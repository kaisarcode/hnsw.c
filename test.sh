#!/bin/sh
# test.sh
# Summary: Validation suite for hnsw approximate vector indexing and top-K search.
# Author:  KaisarCode
# Website: https://kaisarcode.com
# License: https://www.gnu.org/licenses/gpl-3.0.html

# Prints a failure message.
# @param message Failure explanation.
# @return 1.
kc_test_fail() {
    printf '[FAIL] %s\n' "$1"
    return 1
}

# Prints a success message.
# @param message Success label.
# @return 0.
kc_test_pass() {
    printf '[PASS] %s\n' "$1"
    return 0
}

# Verifies that the hnsw binary exists and is executable.
# @return Status code.
kc_test_check_binary() {
    if [ ! -x "./hnsw" ]; then
        printf '%s\n' '[ERROR] hnsw binary not found. Please compile first.'
        return 1
    fi

    return 0
}

# Writes a sample dataset to a file.
# @param path Destination file path.
# @return Status code.
kc_test_write_dataset() {
    dataset_path=$1

    printf '%s\n' '# id x y z' >"$dataset_path" || return 1
    printf '%s\n' 'red 1 0 0' >>"$dataset_path" || return 1
    printf '%s\n' 'green 0 1 0' >>"$dataset_path" || return 1
    printf '%s\n' 'blue 0 0 1' >>"$dataset_path" || return 1
    printf '%s\n' 'pink 0.8 0.2 0' >>"$dataset_path" || return 1
    return 0
}

# Executes a test case and compares the resulting IDs.
# @param label Test case description.
# @param expected_ids Expected space-separated identifiers.
# @param ... Command and arguments to execute.
# @return Status code.
kc_test_run_case() {
    label=$1
    expected_ids=$2
    shift 2

    actual_ids=$("$@" | cut -d: -f1 | xargs) || {
        kc_test_fail "$label"
        return 1
    }

    if [ "$actual_ids" = "$expected_ids" ]; then
        kc_test_pass "$label"
        return 0
    fi

    kc_test_fail "$label"
    printf '%s\n' "Expected IDs: $expected_ids"
    printf '%s\n' "Actual IDs: $actual_ids"
    return 1
}

# Orchestrates the validation suite.
# @return Status code.
kc_test_main() {
    temp_dir=''
    dataset_path=''
    failed=0

    kc_test_check_binary || exit 1

    temp_dir=$(mktemp -d)
    dataset_path="$temp_dir/vectors.txt"

    trap 'rm -rf "$temp_dir"' EXIT INT HUP TERM

    kc_test_write_dataset "$dataset_path" || exit 1

    kc_test_run_case \
        'cosine approximate top-2' \
        'red pink' \
        ./hnsw -d 3 -i "$dataset_path" -q '1 0 0' -k 2 -m cosine || failed=$((failed + 1))

    kc_test_run_case \
        'inner-product approximate top-1' \
        'red' \
        ./hnsw -d 3 -i "$dataset_path" -q '0.7 0.1 0' -k 1 -m inner || failed=$((failed + 1))

    kc_test_run_case \
        'l2 approximate top-2' \
        'pink red' \
        ./hnsw -d 3 -i "$dataset_path" -q '0.7 0.1 0' -k 2 -m l2 || failed=$((failed + 1))

    if [ "$failed" -eq 0 ]; then
        printf '%s\n' '[SUCCESS] All hnsw HNSW validation tests passed!'
        return 0
    fi

    printf '[FAILURE] %s tests failed.\n' "$failed"
    return 1
}

kc_test_main
