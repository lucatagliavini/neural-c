#!/usr/bin/env bash

set -euo pipefail

native_executable=${1:?usage: test_cross_runtime.sh <native-executable> <ppc64le-executable>}
ppc64le_executable=${2:?usage: test_cross_runtime.sh <native-executable> <ppc64le-executable>}
qemu=${PPC64LE_QEMU:-qemu-ppc64le}
sysroot=${PPC64LE_SYSROOT:-/usr/powerpc64le-linux-gnu}
work_dir=build/tests/cross-runtime
xor_inputs=(0 0 0 1 1 0 1 1)
active_process_id=

native_command=("$native_executable")
ppc64le_command=("$qemu" -L "$sysroot" "$ppc64le_executable")

cleanup() {
    if [[ -n $active_process_id ]]; then
        kill -TERM "$active_process_id" 2>/dev/null || true
        wait "$active_process_id" 2>/dev/null || true
    fi
    if [[ ${KEEP_CROSS_RUNTIME_ARTIFACTS:-0} != 1 ]]; then
        rm -rf -- "$work_dir"
    fi
}
trap cleanup EXIT

if [[ ! -x $native_executable ]]; then
    printf 'cross-runtime: native executable not found: %s\n' \
        "$native_executable" >&2
    exit 1
fi
if [[ ! -x $ppc64le_executable ]]; then
    printf 'cross-runtime: ppc64le executable not found: %s\n' \
        "$ppc64le_executable" >&2
    exit 1
fi
if ! command -v "$qemu" >/dev/null 2>&1; then
    printf 'cross-runtime: QEMU executable not found: %s\n' "$qemu" >&2
    exit 1
fi
if [[ ! -r $sysroot/lib/ld64.so.2 ]]; then
    printf 'cross-runtime: ppc64le loader not found under %s\n' \
        "$sysroot" >&2
    exit 1
fi

run_native() {
    "${native_command[@]}" "$@"
}

run_ppc64le() {
    "${ppc64le_command[@]}" "$@"
}

prepare_project() {
    local directory=$1

    mkdir -p "$directory"
    cp projects/xor/model.txt "$directory/model.txt"
    cp projects/xor/project.conf "$directory/project.conf"
    cp projects/xor/train.txt "$directory/train.txt"
}

assert_xor_inspection() {
    local path=$1

    grep -q \
        '^Model digest: sha256:8da5a1f53fc59cabe5e685895a6e254d6b362ff4704f3d6b7325d98b7e4cc0d5$' \
        "$path"
    grep -q \
        '^Dataset digest: sha256:df97fb50ab811253bc1ebfceb93dea8679f5c337bb9995f384f47a7936991275$' \
        "$path"
    grep -q \
        '^Training digest: sha256:08778dfaba9557e47d924f3010e4b972935e718b994c3a65eb2f0aac5de4ab6d$' \
        "$path"
    grep -q '^Validation: OK$' "$path"
}

assert_xor_prediction() {
    local path=$1

    grep -q '^neural-c predictions 1$' "$path"
    grep -q '^completed_epochs 10000$' "$path"
    grep -q '^samples 4$' "$path"
    grep -q '^inputs 2$' "$path"
    grep -q '^outputs 1$' "$path"
    grep -q '^end$' "$path"
    awk '
        $1 == "sample" && $2 == 0 { ok0 = ($3 < 0.2) }
        $1 == "sample" && $2 == 1 { ok1 = ($3 > 0.8) }
        $1 == "sample" && $2 == 2 { ok2 = ($3 > 0.8) }
        $1 == "sample" && $2 == 3 { ok3 = ($3 < 0.2) }
        END { exit !(ok0 && ok1 && ok2 && ok3) }
    ' "$path"
}

compare_prediction_documents() {
    local reference=$1
    local candidate=$2

    if cmp -s "$reference" "$candidate"; then
        return 0
    fi
    awk '
        FNR == NR {
            reference[FNR] = $0
            reference_fields[FNR] = NF
            lines = FNR
            next
        }
        FNR > lines || NF != reference_fields[FNR] { failed = 1; next }
        $1 != "sample" {
            if ($0 != reference[FNR]) failed = 1
            next
        }
        {
            split(reference[FNR], expected)
            if ($1 != expected[1] || $2 != expected[2]) {
                failed = 1
                next
            }
            for (field = 3; field <= NF; field++) {
                difference = $field - expected[field]
                if (difference < 0) difference = -difference
                magnitude = $field
                if (magnitude < 0) magnitude = -magnitude
                expected_magnitude = expected[field]
                if (expected_magnitude < 0) {
                    expected_magnitude = -expected_magnitude
                }
                if (expected_magnitude > magnitude) {
                    magnitude = expected_magnitude
                }
                if (difference > 1e-14 && difference > 1e-12 * magnitude) {
                    failed = 1
                }
            }
        }
        END { if (FNR != lines || failed) exit 1 }
    ' "$reference" "$candidate"
}

compare_scalars() {
    local reference=$1
    local candidate=$2

    awk -v reference="$reference" -v candidate="$candidate" 'BEGIN {
        difference = reference - candidate
        if (difference < 0) difference = -difference
        magnitude = reference
        if (magnitude < 0) magnitude = -magnitude
        candidate_magnitude = candidate
        if (candidate_magnitude < 0) candidate_magnitude = -candidate_magnitude
        if (candidate_magnitude > magnitude) magnitude = candidate_magnitude
        exit difference > 1e-14 && difference > 1e-12 * magnitude
    }'
}

wait_for_training_lock() {
    local directory=$1
    local process_id=$2
    local expected_path
    local descriptor
    local descriptor_path
    local attempt

    expected_path=$(realpath -m "$directory/.neural-c.lock")
    for ((attempt = 0; attempt < 1000; attempt++)); do
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

interrupt_training() {
    local runtime=$1
    local directory=$2
    local process_id
    local status

    if [[ $runtime == native ]]; then
        "${native_command[@]}" train "$directory" --threads 4 \
            >"$directory/stdout.txt" 2>"$directory/stderr.txt" &
    else
        "${ppc64le_command[@]}" train "$directory" --threads 4 \
            >"$directory/stdout.txt" 2>"$directory/stderr.txt" &
    fi
    process_id=$!
    active_process_id=$process_id
    wait_for_training_lock "$directory" "$process_id"
    kill -INT "$process_id"
    set +e
    wait "$process_id"
    status=$?
    set -e
    active_process_id=
    [[ $status -eq 130 ]]
    grep -q 'checkpoint saved' "$directory/stderr.txt"
    [[ -s $directory/checkpoint.txt ]]
    [[ ! -e $directory/weights.txt ]]
}

rm -rf -- "$work_dir"
mkdir -p "$work_dir"

native_dir=$work_dir/native
ppc64le_dir=$work_dir/ppc64le
native_to_ppc64le_dir=$work_dir/native-to-ppc64le
ppc64le_to_native_dir=$work_dir/ppc64le-to-native
native_data_dir=$work_dir/native-data
ppc64le_data_dir=$work_dir/ppc64le-data
prepare_project "$native_dir"
prepare_project "$ppc64le_dir"
prepare_project "$native_to_ppc64le_dir"
prepare_project "$ppc64le_to_native_dir"

printf 'Cross-runtime: validating canonical project inspection\n'
run_native inspect "$native_dir" >"$work_dir/native.inspect"
run_ppc64le inspect "$ppc64le_dir" >"$work_dir/ppc64le.inspect"
assert_xor_inspection "$work_dir/native.inspect"
assert_xor_inspection "$work_dir/ppc64le.inspect"

printf 'Cross-runtime: training native and ppc64le baselines\n'
run_native train "$native_dir" --threads 4 >"$work_dir/native.train"
run_ppc64le train "$ppc64le_dir" --threads 4 >"$work_dir/ppc64le.train"
grep -q '^Training complete: 10000 epochs, loss ' "$work_dir/native.train"
grep -q '^Training complete: 10000 epochs, loss ' "$work_dir/ppc64le.train"
native_loss=$(awk '{ value = $6; sub(/,$/, "", value); print value }' \
    "$work_dir/native.train")
ppc64le_loss=$(awk '{ value = $6; sub(/,$/, "", value); print value }' \
    "$work_dir/ppc64le.train")
compare_scalars "$native_loss" "$ppc64le_loss"

run_native predict "$native_dir" "${xor_inputs[@]}" --threads 1 \
    >"$work_dir/native.predict.1"
run_native predict "$native_dir" "${xor_inputs[@]}" --threads 4 \
    >"$work_dir/native.predict.4"
run_ppc64le predict "$ppc64le_dir" "${xor_inputs[@]}" --threads 1 \
    >"$work_dir/ppc64le.predict.1"
run_ppc64le predict "$ppc64le_dir" "${xor_inputs[@]}" --threads 4 \
    >"$work_dir/ppc64le.predict.4"
cmp "$work_dir/native.predict.1" "$work_dir/native.predict.4"
cmp "$work_dir/ppc64le.predict.1" "$work_dir/ppc64le.predict.4"
assert_xor_prediction "$work_dir/native.predict.1"
assert_xor_prediction "$work_dir/ppc64le.predict.1"
compare_prediction_documents \
    "$work_dir/native.predict.1" "$work_dir/ppc64le.predict.1"

printf 'Cross-runtime: loading native weights under ppc64le\n'
run_ppc64le predict "$native_dir" "${xor_inputs[@]}" --threads 3 \
    >"$work_dir/native-weights.ppc64le-predict"
compare_prediction_documents \
    "$work_dir/native.predict.1" "$work_dir/native-weights.ppc64le-predict"

printf 'Cross-runtime: loading ppc64le weights under native execution\n'
run_native predict "$ppc64le_dir" "${xor_inputs[@]}" --threads 3 \
    >"$work_dir/ppc64le-weights.native-predict"
compare_prediction_documents \
    "$work_dir/ppc64le.predict.1" "$work_dir/ppc64le-weights.native-predict"

printf 'Cross-runtime: resuming a native checkpoint under ppc64le\n'
interrupt_training native "$native_to_ppc64le_dir"
run_ppc64le train "$native_to_ppc64le_dir" --resume --threads 3 \
    >"$work_dir/native-to-ppc64le.resume"
[[ -s $native_to_ppc64le_dir/weights.txt ]]
[[ ! -e $native_to_ppc64le_dir/checkpoint.txt ]]
run_native predict "$native_to_ppc64le_dir" "${xor_inputs[@]}" --threads 2 \
    >"$work_dir/native-to-ppc64le.native-predict"
run_ppc64le predict "$native_to_ppc64le_dir" "${xor_inputs[@]}" --threads 2 \
    >"$work_dir/native-to-ppc64le.ppc64le-predict"
assert_xor_prediction "$work_dir/native-to-ppc64le.native-predict"
compare_prediction_documents \
    "$work_dir/native-to-ppc64le.native-predict" \
    "$work_dir/native-to-ppc64le.ppc64le-predict"

printf 'Cross-runtime: resuming a ppc64le checkpoint under native execution\n'
interrupt_training ppc64le "$ppc64le_to_native_dir"
run_native train "$ppc64le_to_native_dir" --resume --threads 3 \
    >"$work_dir/ppc64le-to-native.resume"
[[ -s $ppc64le_to_native_dir/weights.txt ]]
[[ ! -e $ppc64le_to_native_dir/checkpoint.txt ]]
run_native predict "$ppc64le_to_native_dir" "${xor_inputs[@]}" --threads 2 \
    >"$work_dir/ppc64le-to-native.native-predict"
run_ppc64le predict "$ppc64le_to_native_dir" "${xor_inputs[@]}" --threads 2 \
    >"$work_dir/ppc64le-to-native.ppc64le-predict"
assert_xor_prediction "$work_dir/ppc64le-to-native.native-predict"
compare_prediction_documents \
    "$work_dir/ppc64le-to-native.native-predict" \
    "$work_dir/ppc64le-to-native.ppc64le-predict"

printf 'Cross-runtime: exchanging preprocessing-bound native weights\n'
run_native init "$native_data_dir" --inputs 4 --layer 3:softmax \
    --epochs 2 --checkpoint-interval 0 >/dev/null
run_native import-csv "$native_data_dir" tests/fixtures/iris-small.csv \
    --schema tests/fixtures/iris-schema.txt --validation-ratio 0.16666666666666666 \
    --test-ratio 0.16666666666666666 --split-seed 42 \
    --normalization standardize --missing mean >/dev/null
run_native train "$native_data_dir" --threads 2 >/dev/null
run_native predict "$native_data_dir" 5.1 3.5 1.4 '?' --threads 1 \
    >"$work_dir/native-data.native-predict"
run_ppc64le predict "$native_data_dir" 5.1 3.5 1.4 '?' --threads 2 \
    >"$work_dir/native-data.ppc64le-predict"
compare_prediction_documents \
    "$work_dir/native-data.native-predict" \
    "$work_dir/native-data.ppc64le-predict"

printf 'Cross-runtime: exchanging preprocessing-bound ppc64le weights\n'
run_ppc64le init "$ppc64le_data_dir" --inputs 4 --layer 3:softmax \
    --epochs 2 --checkpoint-interval 0 >/dev/null
run_ppc64le import-csv "$ppc64le_data_dir" tests/fixtures/iris-small.csv \
    --schema tests/fixtures/iris-schema.txt --validation-ratio 0.16666666666666666 \
    --test-ratio 0.16666666666666666 --split-seed 42 \
    --normalization minmax --missing mean >/dev/null
run_ppc64le train "$ppc64le_data_dir" --threads 2 >/dev/null
run_ppc64le predict "$ppc64le_data_dir" 6.4 3.2 4.5 1.5 --threads 1 \
    >"$work_dir/ppc64le-data.ppc64le-predict"
run_native predict "$ppc64le_data_dir" 6.4 3.2 4.5 1.5 --threads 2 \
    >"$work_dir/ppc64le-data.native-predict"
compare_prediction_documents \
    "$work_dir/ppc64le-data.ppc64le-predict" \
    "$work_dir/ppc64le-data.native-predict"

printf 'All cross-runtime tests passed\n'
