#!/usr/bin/env bash

set -euo pipefail

executable=${1:?usage: test_cli.sh <neural-c-executable>}
project_dir="build/tests/cli-init-project"

cleanup() {
    rm -f -- \
        "$project_dir/model.txt" \
        "$project_dir/project.conf" \
        "$project_dir/train.txt" \
        "$project_dir/weights.txt" \
        "$project_dir/checkpoint.txt"
    rmdir -- "$project_dir" 2>/dev/null || true
}
trap cleanup EXIT
cleanup

"$executable" init "$project_dir" \
    --inputs 3 \
    --layer 4:leaky_relu:alpha=0.01 \
    --layer 2:sigmoid \
    --epochs 250 \
    --learning-rate 0.25 \
    --seed 7

grep -q '^input 3$' "$project_dir/model.txt"
grep -q '^dense 4 leaky_relu alpha=0.01$' "$project_dir/model.txt"
grep -q '^dense 2 sigmoid$' "$project_dir/model.txt"
grep -q '^epochs 250$' "$project_dir/project.conf"

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

inspect_output=$("$executable" inspect projects/xor)
grep -Eq '^Model digest: sha256:[0-9a-f]{64}$' <<<"$inspect_output"
grep -Eq '^Dataset digest: sha256:[0-9a-f]{64}$' <<<"$inspect_output"
grep -Eq '^Training digest: sha256:[0-9a-f]{64}$' <<<"$inspect_output"

set +e
resume_output=$("$executable" train "$project_dir" --resume 2>&1)
resume_status=$?
set -e
[[ $resume_status -eq 3 ]]
grep -q "training mode 'resume' is not implemented yet" <<<"$resume_output"

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
threads_output=$("$executable" train "$project_dir" --threads 4 2>&1)
threads_status=$?
set -e
[[ $threads_status -eq 3 ]]
grep -q 'threads: 4' <<<"$threads_output"

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
