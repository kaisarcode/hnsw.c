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
    printf '\033[31m[FAIL]\033[0m %s\n' "$1"
    return 1
}

# Prints a success message.
# @param message Success label.
# @return 0.
kc_test_pass() {
    printf '\033[32m[PASS]\033[0m %s\n' "$1"
    return 0
}

# Verifies that the hnsw binary exists and is executable.
# @return Status code.
kc_test_check_binary() {
    if [ ! -x "./bin/x86_64/linux/hnsw" ]; then
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
    return 1
}

# Verifies API state transitions.
# @return Status code.
kc_test_api_state() {
    label='API state transition (ESTATE)'
    test_src='api_test.c'
    test_bin='./api_test'

    {
        printf '#include "hnsw.h"\n#include <stdio.h>\nint main() {\n'
        printf '  kc_hnsw_t *hnsw = kc_hnsw_open(3, KC_HNSW_METRIC_COSINE);\n'
        printf '  float v[] = {1, 0, 0};\n'
        printf '  kc_hnsw_add(hnsw, "1", v);\n'
        printf '  kc_hnsw_add(hnsw, "2", v);\n'
        printf '  kc_hnsw_add(hnsw, "3", v);\n'
        printf '  kc_hnsw_result_t res[3];\n'
        printf '  int rc = kc_hnsw_search(hnsw, v, 3, 0.0, res);\n'
        printf '  if (rc != %d) { fprintf(stderr, "Expected ESTATE\\n"); return 1; }\n' "-3"
        printf '  kc_hnsw_build(hnsw);\n'
        printf '  rc = kc_hnsw_search(hnsw, v, 3, 0.0, res);\n'
        printf '  if (rc <= 0) { fprintf(stderr, "Expected positive results\\n"); return 1; }\n'
        printf '  kc_hnsw_close(hnsw); return 0;\n}\n'
    } > "$test_src"

    ${CC:-cc} -O3 -o "$test_bin" "$test_src" src/libhnsw.c -lm -lpthread -Isrc >/dev/null 2>&1 || {
        rm -f "$test_src" "$test_bin"
        return 1
    }

    "$test_bin" >/dev/null 2>&1 || {
        rm -f "$test_src" "$test_bin"
        kc_test_fail "$label"
        return 1
    }

    rm -f "$test_src" "$test_bin"
    kc_test_pass "$label"
    return 0
}

# Orchestrates the validation suite.
# @return Status code.
kc_test_main() {
    temp_dir=''
    dataset_path=''
    failed=0

    kc_test_check_binary || exit 1

    BIN="./bin/x86_64/linux/hnsw"

    temp_dir=$(mktemp -d)
    dataset_path="$temp_dir/vectors.txt"

    trap 'rm -rf "$temp_dir"' EXIT INT HUP TERM

    kc_test_write_dataset "$dataset_path" || exit 1

    kc_test_run_case \
        'cosine approximate top-2' \
        'red pink' \
        "$BIN" -d 3 -i "$dataset_path" -q '1 0 0' -k 2 -m cosine || failed=$((failed + 1))

    kc_test_run_case \
        'inner-product approximate top-1' \
        'red' \
        "$BIN" -d 3 -i "$dataset_path" -q '0.7 0.1 0' -k 1 -m inner || failed=$((failed + 1))

    kc_test_run_case \
        'l2 approximate top-2' \
        'pink red' \
        "$BIN" -d 3 -i "$dataset_path" -q '0.7 0.1 0' -k 2 -m l2 || failed=$((failed + 1))

    kc_test_api_state || failed=$((failed + 1))

    if [ "$failed" -eq 0 ]; then
        return 0
    fi

    return 1
}

kc_test_main
