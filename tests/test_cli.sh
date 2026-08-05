#!/usr/bin/env bash

set -euo pipefail

executable=${1:?usage: test_cli.sh <neural-c-executable>}
project_dir="build/tests/cli-init-project"
training_dir="build/tests/cli-training-project"
interrupt_dir="build/tests/cli-interrupt-project"
early_dir="build/tests/cli-early-project"
import_dir="build/tests/cli-import-project"

cleanup() {
    rm -f -- \
        "$project_dir/model.txt" \
        "$project_dir/project.conf" \
        "$project_dir/train.txt" \
        "$project_dir/weights.txt" \
        "$project_dir/checkpoint.txt" \
        "$project_dir/.neural-c.lock"
    rmdir -- "$project_dir" 2>/dev/null || true
    rm -f -- \
        "$training_dir/model.txt" \
        "$training_dir/project.conf" \
        "$training_dir/train.txt" \
        "$training_dir/validation.txt" \
        "$training_dir/test.txt" \
        "$training_dir/history.txt" \
        "$training_dir/weights.txt" \
        "$training_dir/weights.before" \
        "$training_dir/checkpoint.txt" \
        "$training_dir/stdout.txt" \
        "$training_dir/stderr.txt" \
        "$training_dir/.neural-c.lock"
    rmdir -- "$training_dir" 2>/dev/null || true
    rm -f -- \
        "$interrupt_dir/model.txt" \
        "$interrupt_dir/project.conf" \
        "$interrupt_dir/train.txt" \
        "$interrupt_dir/weights.txt" \
        "$interrupt_dir/weights.before" \
        "$interrupt_dir/checkpoint.txt" \
        "$interrupt_dir/.neural-c.lock" \
        "$interrupt_dir/stdout.txt" \
        "$interrupt_dir/stderr.txt"
    rmdir -- "$interrupt_dir" 2>/dev/null || true
    rm -f -- \
        "$early_dir/model.txt" \
        "$early_dir/project.conf" \
        "$early_dir/train.txt" \
        "$early_dir/validation.txt" \
        "$early_dir/test.txt" \
        "$early_dir/weights.txt" \
        "$early_dir/checkpoint.txt" \
        "$early_dir/.neural-c.lock"
    rmdir -- "$early_dir" 2>/dev/null || true
    rm -f -- \
        "$import_dir/model.txt" \
        "$import_dir/project.conf" \
        "$import_dir/train.txt" \
        "$import_dir/validation.txt" \
        "$import_dir/test.txt" \
        "$import_dir/preprocessing.txt" \
        "$import_dir/weights.txt" \
        "$import_dir/checkpoint.txt" \
        "$import_dir/.neural-c.lock"
    rmdir -- "$import_dir" 2>/dev/null || true
}
trap cleanup EXIT
cleanup

wait_for_training_lock() {
    local directory=$1
    local process_id=$2
    local attempt
    local descriptor
    local expected_path
    local descriptor_path

    expected_path=$(realpath -m "$directory/.neural-c.lock")

    for ((attempt = 0; attempt < 500; attempt++)); do
        if ! kill -0 "$process_id" 2>/dev/null; then
            return 1
        fi
        for descriptor in "/proc/$process_id/fd/"*; do
            descriptor_path=$(readlink "$descriptor" 2>/dev/null || true)
            if [[ $descriptor_path == "$expected_path" ]]; then
                return 0
            fi
        done
        sleep 0.001
    done
    return 1
}

set +e
invalid_loss_output=$(
    "$executable" init "$project_dir" --inputs 1 --layer 1:linear \
        --loss binary_cross_entropy 2>&1
)
invalid_loss_status=$?
set -e
[[ $invalid_loss_status -eq 2 ]]
grep -q 'binary_cross_entropy requires sigmoid output' \
    <<<"$invalid_loss_output"
[[ ! -e $project_dir ]]

"$executable" init "$project_dir" \
    --inputs 3 \
    --layer 4:leaky_relu:alpha=0.01 \
    --layer 2:sigmoid \
    --epochs 250 \
    --learning-rate 0.25 \
    --seed 7 \
    --checkpoint-interval 0 \
    --batch-size 3 \
    --gradient-clip-norm 0.75 \
    --shuffle

grep -q '^input 3$' "$project_dir/model.txt"
grep -q '^dense 4 leaky_relu alpha=0.01$' "$project_dir/model.txt"
grep -q '^dense 2 sigmoid$' "$project_dir/model.txt"
grep -q '^epochs 250$' "$project_dir/project.conf"
grep -q '^checkpoint_interval 0$' "$project_dir/project.conf"
grep -q '^batch_size 3$' "$project_dir/project.conf"
grep -q '^shuffle 1$' "$project_dir/project.conf"
grep -q '^gradient_clip_norm 0.75$' "$project_dir/project.conf"
printf '0 0 0 -> 0 0\n' >>"$project_dir/train.txt"
configured_inspect=$("$executable" inspect "$project_dir")
grep -q '^Batch size: 3$' <<<"$configured_inspect"
grep -q '^Shuffle: enabled$' <<<"$configured_inspect"
grep -q '^Gradient clip norm: 0.75$' <<<"$configured_inspect"

if "$executable" init "$project_dir" --inputs 1 --layer 1:sigmoid; then
    echo "init unexpectedly overwrote an existing project" >&2
    exit 1
fi

"$executable" init "$project_dir" --inputs 1 --layer 1:tanh --force
grep -q '^input 1$' "$project_dir/model.txt"
grep -q '^dense 1 tanh$' "$project_dir/model.txt"
grep -q '^epochs 10000$' "$project_dir/project.conf"
grep -q '^learning_rate 0.5$' "$project_dir/project.conf"
grep -q '^checkpoint_interval 100$' "$project_dir/project.conf"
grep -q '^batch_size 0$' "$project_dir/project.conf"
grep -q '^shuffle 0$' "$project_dir/project.conf"
grep -q '^gradient_clip_norm 0$' "$project_dir/project.conf"
! grep -q 'threads' "$project_dir/project.conf"

exec {init_lock_fd}<>"$project_dir/.neural-c.lock"
flock -n "$init_lock_fd"
set +e
busy_init_output=$(
    "$executable" init "$project_dir" --inputs 1 --layer 1:sigmoid --force 2>&1
)
busy_init_status=$?
set -e
[[ $busy_init_status -eq 2 ]]
grep -q 'project is busy' <<<"$busy_init_output"
grep -q '^dense 1 tanh$' "$project_dir/model.txt"
flock -u "$init_lock_fd"
exec {init_lock_fd}>&-

inspect_output=$("$executable" inspect projects/xor)
grep -q '^Batch size: full dataset$' <<<"$inspect_output"
grep -q '^Shuffle: disabled$' <<<"$inspect_output"
grep -q '^Gradient clip norm: disabled$' <<<"$inspect_output"
grep -Eq '^Model digest: sha256:[0-9a-f]{64}$' <<<"$inspect_output"
grep -Eq '^Dataset digest: sha256:[0-9a-f]{64}$' <<<"$inspect_output"
grep -Eq '^Training digest: sha256:[0-9a-f]{64}$' <<<"$inspect_output"

exec {inspect_lock_fd}<>"projects/xor/.neural-c.lock"
flock -n "$inspect_lock_fd"
set +e
busy_inspect_output=$("$executable" inspect projects/xor 2>&1)
busy_inspect_status=$?
set -e
[[ $busy_inspect_status -eq 1 ]]
grep -q 'project is busy' <<<"$busy_inspect_output"
flock -u "$inspect_lock_fd"
exec {inspect_lock_fd}>&-

set +e
resume_output=$("$executable" train "$project_dir" --resume 2>&1)
resume_status=$?
set -e
[[ $resume_status -eq 1 ]]
grep -q 'resume requires checkpoint.txt' <<<"$resume_output"

set +e
additional_output=$(
    "$executable" train "$project_dir" --additional-epochs 200 2>&1
)
additional_status=$?
set -e
[[ $additional_status -eq 1 ]]
grep -q 'requires finalized weights.txt' <<<"$additional_output"

set +e
invalid_threads_output=$("$executable" train "$project_dir" --threads 0 2>&1)
invalid_threads_status=$?
set -e
[[ $invalid_threads_status -eq 2 ]]
grep -q 'thread count must be a positive integer' <<<"$invalid_threads_output"

set +e
invalid_report_output=$(
    "$executable" train "$project_dir" --report-interval nope 2>&1
)
invalid_report_status=$?
set -e
[[ $invalid_report_status -eq 2 ]]
grep -q 'report-interval must be a non-negative integer' \
    <<<"$invalid_report_output"

set +e
predict_threads_output=$(
    "$executable" predict "$project_dir" 0 --threads 2 2>&1
)
predict_threads_status=$?
set -e
[[ $predict_threads_status -eq 1 ]]
grep -q 'dataset must contain at least one sample' <<<"$predict_threads_output"

set +e
conflict_output=$(
    "$executable" train "$project_dir" --resume --additional-epochs 10 2>&1
)
conflict_status=$?
set -e
[[ $conflict_status -eq 2 ]]
grep -q 'mutually exclusive' <<<"$conflict_output"

mkdir -p "$training_dir"
cp projects/xor/model.txt "$training_dir/model.txt"
cp projects/xor/project.conf "$training_dir/project.conf"
cp projects/xor/train.txt "$training_dir/train.txt"

exec {training_lock_fd}<>"$training_dir/.neural-c.lock"
flock -n "$training_lock_fd"
set +e
busy_training_output=$("$executable" train "$training_dir" 2>&1)
busy_training_status=$?
set -e
[[ $busy_training_status -eq 1 ]]
grep -q 'project is busy' <<<"$busy_training_output"
[[ ! -e "$training_dir/weights.txt" ]]

"$executable" init "$early_dir" \
    --inputs 2 \
    --layer 2:sigmoid \
    --layer 1:sigmoid \
    --epochs 100 \
    --learning-rate 0.5 \
    --seed 42 \
    --checkpoint-interval 0 \
    --early-stopping-patience 3 \
    --early-stopping-min-delta 100
cp projects/xor/train.txt "$early_dir/train.txt"
cp projects/xor/train.txt "$early_dir/validation.txt"
cp projects/xor/train.txt "$early_dir/test.txt"
early_training_output=$(
    "$executable" train "$early_dir" --report-interval 2 --threads 2 2>&1
)
grep -q '^Training progress: epoch 4/100, loss ' \
    <<<"$early_training_output"
grep -q '^Training complete: 4 epochs, loss ' <<<"$early_training_output"
grep -q '^neural-c weights 2$' "$early_dir/weights.txt"
grep -q '^completed_epochs 4$' "$early_dir/weights.txt"
grep -q '^selected_epoch 1$' "$early_dir/weights.txt"
grep -q '^target_epochs 100$' "$early_dir/weights.txt"
grep -q '^completion early_stopping$' "$early_dir/weights.txt"
early_prediction=$(
    "$executable" predict "$early_dir" 0 1 --threads 2
)
grep -q '^neural-c predictions 2$' <<<"$early_prediction"
grep -q '^completed_epochs 4$' <<<"$early_prediction"
grep -q '^selected_epoch 1$' <<<"$early_prediction"
grep -q '^completion early_stopping$' <<<"$early_prediction"
early_evaluation=$(
    "$executable" evaluate "$early_dir" --dataset test --threads 2
)
grep -q '^neural-c evaluation 2$' <<<"$early_evaluation"
grep -q '^selected_epoch 1$' <<<"$early_evaluation"
early_state=$("$executable" inspect "$early_dir" --state)
grep -q '^Weights completed epochs: 4$' <<<"$early_state"
grep -q '^Weights selected epoch: 1$' <<<"$early_state"
grep -q '^Weights completion: early_stopping$' <<<"$early_state"
early_additional_output=$(
    "$executable" train "$early_dir" --additional-epochs 10 --threads 2
)
grep -q '^Training complete: 7 epochs, loss ' <<<"$early_additional_output"
grep -q '^completed_epochs 7$' "$early_dir/weights.txt"
grep -q '^selected_epoch 1$' "$early_dir/weights.txt"
grep -q '^target_epochs 14$' "$early_dir/weights.txt"
grep -q '^completion early_stopping$' "$early_dir/weights.txt"

set +e
busy_additional_output=$(
    "$executable" train "$training_dir" --additional-epochs 2 2>&1
)
busy_additional_status=$?
set -e
[[ $busy_additional_status -eq 1 ]]
grep -q 'project is busy' <<<"$busy_additional_output"

set +e
busy_predict_output=$(
    "$executable" predict "$training_dir" 0 0 --threads 2 2>&1
)
busy_predict_status=$?
set -e
[[ $busy_predict_status -eq 1 ]]
grep -q 'project is busy' <<<"$busy_predict_output"

mkdir -p "$interrupt_dir"
cp projects/xor/model.txt "$interrupt_dir/model.txt"
cp projects/xor/project.conf "$interrupt_dir/project.conf"
cp projects/xor/train.txt "$interrupt_dir/train.txt"
sed -i 's/^checkpoint_interval .*/checkpoint_interval 0/' \
    "$interrupt_dir/project.conf"

"$executable" train "$interrupt_dir" --threads 4 \
    >"$interrupt_dir/stdout.txt" 2>"$interrupt_dir/stderr.txt" &
interrupt_pid=$!
wait_for_training_lock "$interrupt_dir" "$interrupt_pid"
kill -INT "$interrupt_pid"
set +e
wait "$interrupt_pid"
interrupt_status=$?
set -e
[[ $interrupt_status -eq 130 ]]
grep -q 'signal 2' "$interrupt_dir/stderr.txt"
grep -q 'checkpoint saved' "$interrupt_dir/stderr.txt"
grep -Eq '^completed_epochs [1-9][0-9]*$' \
    "$interrupt_dir/checkpoint.txt"
[[ ! -e "$interrupt_dir/weights.txt" ]]
first_interrupted_epoch=$(
    awk '$1 == "completed_epochs" { print $2 }' \
        "$interrupt_dir/checkpoint.txt"
)

"$executable" train "$interrupt_dir" --resume --threads 2 \
    >"$interrupt_dir/stdout.txt" 2>"$interrupt_dir/stderr.txt" &
terminate_pid=$!
wait_for_training_lock "$interrupt_dir" "$terminate_pid"
kill -TERM "$terminate_pid"
set +e
wait "$terminate_pid"
terminate_status=$?
set -e
[[ $terminate_status -eq 143 ]]
grep -q 'signal 15' "$interrupt_dir/stderr.txt"
grep -q 'checkpoint saved' "$interrupt_dir/stderr.txt"
second_interrupted_epoch=$(
    awk '$1 == "completed_epochs" { print $2 }' \
        "$interrupt_dir/checkpoint.txt"
)
((second_interrupted_epoch > first_interrupted_epoch))
[[ ! -e "$interrupt_dir/weights.txt" ]]

resume_after_signal_output=$(
    "$executable" train "$interrupt_dir" --resume --threads 4
)
grep -q '^Training complete: 10000 epochs, loss ' \
    <<<"$resume_after_signal_output"
[[ -e "$interrupt_dir/weights.txt" ]]
[[ ! -e "$interrupt_dir/checkpoint.txt" ]]

cp "$interrupt_dir/weights.txt" "$interrupt_dir/weights.before"
"$executable" train "$interrupt_dir" --additional-epochs 20000 --threads 4 \
    >"$interrupt_dir/stdout.txt" 2>"$interrupt_dir/stderr.txt" &
refinement_pid=$!
wait_for_training_lock "$interrupt_dir" "$refinement_pid"
kill -INT "$refinement_pid"
set +e
wait "$refinement_pid"
refinement_status=$?
set -e
[[ $refinement_status -eq 130 ]]
grep -q 'checkpoint saved' "$interrupt_dir/stderr.txt"
cmp "$interrupt_dir/weights.before" "$interrupt_dir/weights.txt"
grep -q '^target_epochs 30000$' "$interrupt_dir/checkpoint.txt"
refinement_epoch=$(
    awk '$1 == "completed_epochs" { print $2 }' \
        "$interrupt_dir/checkpoint.txt"
)
((refinement_epoch > 10000 && refinement_epoch < 30000))

baseline_prediction=$(
    "$executable" predict "$interrupt_dir" 0 0 --threads 2
)
grep -q '^completed_epochs 10000$' <<<"$baseline_prediction"

refinement_resume_output=$(
    "$executable" train "$interrupt_dir" --resume --threads 2
)
grep -q '^Training complete: 30000 epochs, loss ' \
    <<<"$refinement_resume_output"
grep -q '^completed_epochs 30000$' "$interrupt_dir/weights.txt"
[[ ! -e "$interrupt_dir/checkpoint.txt" ]]
[[ ! -e "$training_dir/checkpoint.txt" ]]

flock -u "$training_lock_fd"
exec {training_lock_fd}>&-

"$executable" train "$training_dir" --threads 4 --report-interval 2500 \
    --history \
    >"$training_dir/stdout.txt" 2>"$training_dir/stderr.txt"
training_output=$(<"$training_dir/stdout.txt")
grep -q '^Training complete: 10000 epochs, loss ' <<<"$training_output"
grep -q ', workers 4$' <<<"$training_output"
[[ $(grep -c '^Training progress: epoch ' "$training_dir/stderr.txt") -eq 4 ]]
grep -q '^Training progress: epoch 2500/10000, loss .*best .*improvement n/a, relative n/a$' \
    "$training_dir/stderr.txt"
grep -q '^Training progress: epoch 10000/10000, loss .*improvement .*relative ' \
    "$training_dir/stderr.txt"
grep -q '^Training progress: .*gradient norm .*clipped batches ' \
    "$training_dir/stderr.txt"
grep -q '^neural-c weights 1$' "$training_dir/weights.txt"
grep -q '^completed_epochs 10000$' "$training_dir/weights.txt"
[[ ! -e "$training_dir/checkpoint.txt" ]]
grep -q '^neural-c history 1$' "$training_dir/history.txt"
[[ $(grep -c '^epoch ' "$training_dir/history.txt") -eq 4 ]]
grep -q '^epoch 10000 target 10000 loss .* best ' \
    "$training_dir/history.txt"
grep -q '^epoch .*gradient_norm .*clipped_batches ' \
    "$training_dir/history.txt"

state_output=$("$executable" inspect "$training_dir" --state)
grep -q '^Weights state: present$' <<<"$state_output"
grep -q '^Weights completed epochs: 10000$' <<<"$state_output"
grep -q '^Checkpoint state: absent$' <<<"$state_output"

cp "$training_dir/train.txt" "$training_dir/validation.txt"
cp "$training_dir/train.txt" "$training_dir/test.txt"
evaluation_parallel=$(
    "$executable" evaluate "$training_dir" --dataset test --threads 4
)
evaluation_sequential=$(
    "$executable" evaluate "$training_dir" --dataset test --threads 1
)
[[ "$evaluation_parallel" == "$evaluation_sequential" ]]
grep -q '^neural-c evaluation 1$' <<<"$evaluation_parallel"
grep -q '^completed_epochs 10000$' <<<"$evaluation_parallel"
grep -q '^dataset test$' <<<"$evaluation_parallel"
grep -q '^samples 4$' <<<"$evaluation_parallel"
grep -q '^classification yes$' <<<"$evaluation_parallel"
grep -q '^correct 4$' <<<"$evaluation_parallel"
grep -q '^accuracy 1$' <<<"$evaluation_parallel"
grep -q '^confusion 0 2 0$' <<<"$evaluation_parallel"
grep -q '^confusion 1 0 2$' <<<"$evaluation_parallel"
grep -q '^class 0 precision 1 recall 1 f1 1$' <<<"$evaluation_parallel"
grep -q '^class 1 precision 1 recall 1 f1 1$' <<<"$evaluation_parallel"
grep -q '^end$' <<<"$evaluation_parallel"
evaluation_validation=$(
    "$executable" evaluate "$training_dir" --dataset validation --threads 2
)
grep -q '^dataset validation$' <<<"$evaluation_validation"

set +e
invalid_dataset_output=$(
    "$executable" evaluate "$training_dir" --dataset unknown 2>&1
)
invalid_dataset_status=$?
set -e
[[ $invalid_dataset_status -eq 2 ]]
grep -q 'dataset must be train, validation, or test' \
    <<<"$invalid_dataset_output"

additional_training_output=$(
    "$executable" train "$training_dir" --additional-epochs 2 --threads 2
)
grep -q '^Training complete: 10002 epochs, loss ' \
    <<<"$additional_training_output"
grep -q ', workers 2$' <<<"$additional_training_output"
grep -q '^completed_epochs 10002$' "$training_dir/weights.txt"
[[ ! -e "$training_dir/checkpoint.txt" ]]

prediction_output=$(
    "$executable" predict "$training_dir" 0 0 0 1 1 0 1 1 --threads 4
)
prediction_sequential=$(
    "$executable" predict "$training_dir" 0 0 0 1 1 0 1 1 --threads 1
)
[[ "$prediction_output" == "$prediction_sequential" ]]

prediction_document="build/tests/prediction-input-v1.txt"
cat >"$prediction_document" <<'EOF'
neural-c inputs 1
samples 4
inputs 2
sample 0 0 0
sample 1 0 1
sample 2 1 0
sample 3 1 1
end
EOF
prediction_file=$(
    "$executable" predict "$training_dir" --input "$prediction_document" \
        --batch-size 3 --threads 4
)
prediction_stdin=$(
    "$executable" predict "$training_dir" --input - --batch-size 1 \
        --threads 1 <"$prediction_document"
)
[[ "$prediction_file" == "$prediction_output" ]]
[[ "$prediction_stdin" == "$prediction_output" ]]
printf '%s\n' 'unexpected' >>"$prediction_document"
set +e
malformed_document_output=$(
    "$executable" predict "$training_dir" --input "$prediction_document" \
        --batch-size 2 2>"build/tests/prediction-input-error.txt"
)
malformed_document_status=$?
set -e
[[ $malformed_document_status -eq 1 ]]
[[ -z $malformed_document_output ]]
grep -q 'unexpected content after end' build/tests/prediction-input-error.txt
rm -f -- "$prediction_document" build/tests/prediction-input-error.txt
grep -q '^neural-c predictions 1$' <<<"$prediction_output"
grep -q '^completed_epochs 10002$' <<<"$prediction_output"
grep -q '^samples 4$' <<<"$prediction_output"
grep -q '^inputs 2$' <<<"$prediction_output"
grep -q '^outputs 1$' <<<"$prediction_output"
grep -q '^end$' <<<"$prediction_output"
awk '
    $1 == "sample" && $2 == 0 { ok0 = ($3 < 0.2) }
    $1 == "sample" && $2 == 1 { ok1 = ($3 > 0.8) }
    $1 == "sample" && $2 == 2 { ok2 = ($3 > 0.8) }
    $1 == "sample" && $2 == 3 { ok3 = ($3 < 0.2) }
    END { exit !(ok0 && ok1 && ok2 && ok3) }
' <<<"$prediction_output"

set +e
incomplete_prediction=$(
    "$executable" predict "$training_dir" 0 --threads 2 2>&1
)
incomplete_prediction_status=$?
set -e
[[ $incomplete_prediction_status -eq 2 ]]
grep -q 'complete samples of 2 inputs' <<<"$incomplete_prediction"

set +e
invalid_prediction=$(
    "$executable" predict "$training_dir" 0 nan --threads 2 2>&1
)
invalid_prediction_status=$?
set -e
[[ $invalid_prediction_status -eq 2 ]]
grep -q 'invalid prediction input 1' <<<"$invalid_prediction"

cp "$training_dir/weights.txt" "$training_dir/weights.before"
set +e
repeat_output=$("$executable" train "$training_dir" --threads 2 2>&1)
repeat_status=$?
set -e
[[ $repeat_status -eq 1 ]]
grep -q 'weights.txt already exists' <<<"$repeat_output"
cmp "$training_dir/weights.before" "$training_dir/weights.txt"

rm -f -- "$training_dir/weights.txt" "$training_dir/weights.before"
: >"$training_dir/checkpoint.txt"
set +e
checkpoint_output=$("$executable" train "$training_dir" 2>&1)
checkpoint_status=$?
set -e
[[ $checkpoint_status -eq 1 ]]
grep -q 'checkpoint.txt already exists' <<<"$checkpoint_output"
[[ ! -e "$training_dir/weights.txt" ]]

"$executable" init "$import_dir" --inputs 4 --layer 3:softmax \
    --epochs 2 --loss categorical_cross_entropy --checkpoint-interval 0
import_output=$(
    "$executable" import-csv "$import_dir" tests/fixtures/iris-small.csv \
        --schema tests/fixtures/iris-schema.txt \
        --validation-ratio 0.16666666666666666 \
        --test-ratio 0.16666666666666666 --split-seed 42 \
        --normalization standardize --missing mean
)
grep -q '^CSV import complete: 18 samples, train 12, validation 3, test 3, stratified yes$' \
    <<<"$import_output"
grep -q '^neural-c preprocessing 2$' "$import_dir/preprocessing.txt"
grep -q '^normalization standardize$' "$import_dir/preprocessing.txt"
grep -q '^missing mean$' "$import_dir/preprocessing.txt"
grep -q '^split_algorithm global_largest_remainder_v1$' \
    "$import_dir/preprocessing.txt"
grep -q '^loss categorical_cross_entropy$' "$import_dir/project.conf"
[[ $(grep -c ' -> ' "$import_dir/train.txt") -eq 12 ]]
[[ $(grep -c ' -> ' "$import_dir/validation.txt") -eq 3 ]]
[[ $(grep -c ' -> ' "$import_dir/test.txt") -eq 3 ]]
"$executable" inspect "$import_dir" >/dev/null
"$executable" train "$import_dir" --threads 2 >/dev/null
import_evaluation=$(
    "$executable" evaluate "$import_dir" --dataset test --threads 2
)
grep -q '^classification yes$' <<<"$import_evaluation"
grep -q '^loss ' <<<"$import_evaluation"
import_prediction=$(
    "$executable" predict "$import_dir" 5.1 3.5 1.4 '?' --threads 2
)
grep -q '^samples 1$' <<<"$import_prediction"
grep -q '^outputs 3$' <<<"$import_prediction"
grep -q '^end$' <<<"$import_prediction"
set +e
reimport_output=$(
    "$executable" import-csv "$import_dir" tests/fixtures/iris-small.csv \
        --schema tests/fixtures/iris-schema.txt --missing mean 2>&1
)
reimport_status=$?
set -e
[[ $reimport_status -eq 1 ]]
grep -q 'would invalidate weights/checkpoint' <<<"$reimport_output"
