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

# Detects the artifact architecture for the current machine.
# @return Architecture name on stdout.
kc_test_arch() {
    case "$(uname -m)" in
        x86_64 | amd64)
            printf '%s\n' "x86_64"
            ;;
        aarch64 | arm64)
            printf '%s\n' "aarch64"
            ;;
        armv7l | armv7)
            printf '%s\n' "armv7"
            ;;
        i386 | i486 | i586 | i686)
            printf '%s\n' "i686"
            ;;
        ppc64le | powerpc64le)
            printf '%s\n' "powerpc64le"
            ;;
        *)
            uname -m
            ;;
    esac
}

# Detects the artifact platform for the current machine.
# @return Platform name on stdout.
kc_test_platform() {
    case "$(uname -s)" in
        Linux)
            printf '%s\n' "linux"
            ;;
        *)
            uname -s | tr '[:upper:]' '[:lower:]'
            ;;
    esac
}

# Returns the CLI path for the current architecture and platform.
# @return CLI path on stdout.
kc_test_binary_path() {
    printf './bin/%s/%s/hnsw\n' "$(kc_test_arch)" "$(kc_test_platform)"
}

# Verifies that the hnsw binary exists and is executable.
# @return Status code.
kc_test_check_binary() {
    if [ ! -x "$BIN" ]; then
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

# Executes a CLI success case.
# @param label Test case description.
# @param ... Command and arguments to execute.
# @return Status code.
kc_test_run_success_case() {
    label=$1
    shift

    if "$@" >/dev/null 2>&1; then
        kc_test_pass "$label"
        return 0
    fi

    kc_test_fail "$label"
    return 1
}

# Executes a CLI failure case.
# @param label Test case description.
# @param ... Command and arguments to execute.
# @return Status code.
kc_test_run_fail_case() {
    label=$1
    shift

    if "$@" >/dev/null 2>&1; then
        kc_test_fail "$label"
        return 1
    fi

    kc_test_pass "$label"
    return 0
}

# Orchestrates the validation suite.
# @return Status code.
kc_test_main() {
    temp_dir=''
    dataset_path=''
    failed=0

    BIN=$(kc_test_binary_path)

    kc_test_check_binary || exit 1

    temp_dir=$(mktemp -d)
    dataset_path="$temp_dir/vectors.txt"

    trap 'rm -rf "$temp_dir"' EXIT INT HUP TERM

    kc_test_write_dataset "$dataset_path" || exit 1

    kc_test_run_success_case \
        'help exits successfully' \
        "$BIN" --help || failed=$((failed + 1))

    kc_test_run_success_case \
        'version exits successfully' \
        "$BIN" --version || failed=$((failed + 1))

    kc_test_run_case \
        'cosine approximate top-2' \
        'red pink' \
        "$BIN" -d 3 -i "$dataset_path" -q '1 0 0' -k 2 -m cosine || failed=$((failed + 1))

    kc_test_run_case \
        'cosine comma query top-1' \
        'red' \
        "$BIN" -d 3 -i "$dataset_path" -q '1,0,0' -k 1 -m cosine || failed=$((failed + 1))

    kc_test_run_case \
        'long option cosine top-1' \
        'red' \
        "$BIN" --dim 3 --input "$dataset_path" --query '1 0 0' --top 1 --metric cosine || failed=$((failed + 1))

    kc_test_run_case \
        'cosine threshold filters low scores' \
        'red pink' \
        "$BIN" -d 3 -i "$dataset_path" -q '1 0 0' -k 4 -m cosine -t 0.8 || failed=$((failed + 1))

    kc_test_run_case \
        'inner-product approximate top-1' \
        'red' \
        "$BIN" -d 3 -i "$dataset_path" -q '0.7 0.1 0' -k 1 -m inner || failed=$((failed + 1))

    kc_test_run_case \
        'l2 approximate top-2' \
        'pink red' \
        "$BIN" -d 3 -i "$dataset_path" -q '0.7 0.1 0' -k 2 -m l2 || failed=$((failed + 1))

    kc_test_run_fail_case \
        'query dimension mismatch fails' \
        "$BIN" -d 3 -i "$dataset_path" -q '1 0' || failed=$((failed + 1))

    if [ "$failed" -eq 0 ]; then
        return 0
    fi

    return 1
}

kc_test_main
