#!/usr/bin/env bash

set -euo pipefail

executable=${1:?usage: test_cli.sh <neural-c-executable>}
project_dir="build/tests/cli-init-project"
training_dir="build/tests/cli-training-project"
interrupt_dir="build/tests/cli-interrupt-project"

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
        "$training_dir/weights.txt" \
        "$training_dir/weights.before" \
        "$training_dir/checkpoint.txt" \
        "$training_dir/.neural-c.lock"
    rmdir -- "$training_dir" 2>/dev/null || true
    rm -f -- \
        "$interrupt_dir/model.txt" \
        "$interrupt_dir/project.conf" \
        "$interrupt_dir/train.txt" \
        "$interrupt_dir/weights.txt" \
        "$interrupt_dir/checkpoint.txt" \
        "$interrupt_dir/.neural-c.lock" \
        "$interrupt_dir/stdout.txt" \
        "$interrupt_dir/stderr.txt"
    rmdir -- "$interrupt_dir" 2>/dev/null || true
}
trap cleanup EXIT
cleanup

wait_for_training_lock() {
    local directory=$1
    local process_id=$2
    local attempt
    local probe_fd

    for ((attempt = 0; attempt < 500; attempt++)); do
        if ! kill -0 "$process_id" 2>/dev/null; then
            return 1
        fi
        exec {probe_fd}<>"$directory/.neural-c.lock"
        if flock -n "$probe_fd"; then
            flock -u "$probe_fd"
            exec {probe_fd}>&-
        else
            exec {probe_fd}>&-
            return 0
        fi
        sleep 0.01
    done
    return 1
}

"$executable" init "$project_dir" \
    --inputs 3 \
    --layer 4:leaky_relu:alpha=0.01 \
    --layer 2:sigmoid \
    --epochs 250 \
    --learning-rate 0.25 \
    --seed 7 \
    --checkpoint-interval 0

grep -q '^input 3$' "$project_dir/model.txt"
grep -q '^dense 4 leaky_relu alpha=0.01$' "$project_dir/model.txt"
grep -q '^dense 2 sigmoid$' "$project_dir/model.txt"
grep -q '^epochs 250$' "$project_dir/project.conf"
grep -q '^checkpoint_interval 0$' "$project_dir/project.conf"

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
[[ $additional_status -eq 3 ]]
grep -q "training mode 'additional' is not implemented yet" \
    <<<"$additional_output"

set +e
invalid_threads_output=$("$executable" train "$project_dir" --threads 0 2>&1)
invalid_threads_status=$?
set -e
[[ $invalid_threads_status -eq 2 ]]
grep -q 'thread count must be a positive integer' <<<"$invalid_threads_output"

set +e
predict_threads_output=$(
    "$executable" predict "$project_dir" 0 --threads 2 2>&1
)
predict_threads_status=$?
set -e
[[ $predict_threads_status -eq 3 ]]
grep -q 'threads: 2' <<<"$predict_threads_output"

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
[[ ! -e "$training_dir/checkpoint.txt" ]]
flock -u "$training_lock_fd"
exec {training_lock_fd}>&-

training_output=$("$executable" train "$training_dir" --threads 4)
grep -q '^Training complete: 10000 epochs, loss ' <<<"$training_output"
grep -q ', workers 4$' <<<"$training_output"
grep -q '^neural-c weights 1$' "$training_dir/weights.txt"
grep -q '^completed_epochs 10000$' "$training_dir/weights.txt"
[[ ! -e "$training_dir/checkpoint.txt" ]]

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
