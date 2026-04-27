/**
 * hnsw.c - HNSW Vector Search
 * Summary: CLI for HNSW-based approximate nearest neighbor search.
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "hnsw.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HNSW_LINE_CAP 16384
#define HNSW_QUERY_CAP 4096
#define HNSW_RESULT_CAP 256
#define HNSW_VERSION "0.1.0"

/**
 * Prints the compact command help.
 * @param name Program executable name.
 * @return No return value.
 */
static void hnsw_help(const char *name) {
    printf("Usage:\n");
    printf("  %s -d <dim> -i <dataset> -q <values> [options]\n\n", name);
    printf("Options:\n");
    printf("  --dim, -d <n>        Vector dimension\n");
    printf("  --input, -i <path>   Dataset file with: id v1 v2 ... vN\n");
    printf("  --query, -q <text>   Query vector values separated by spaces or commas\n");
    printf("  --top, -k <n>        Maximum number of results\n");
    printf("  --threshold, -t <n>  Minimum score or maximum distance\n");
    printf("  --metric, -m <name>  cosine | inner | l2\n");
    printf("  --help, -h           Show help\n");
    printf("  --version, -v        Show version\n\n");
    printf("Examples:\n");
    printf("  %s -d 3 -i vectors.txt -q \"1 0 0\"\n", name);
    printf("  %s -d 2 -i points.txt -q \"0.1,0.2\" -m l2 -k 5\n", name);
}

/**
 * Prints the binary version.
 * @return No return value.
 */
static void hnsw_version(void) {
    printf("hnsw %s\n", HNSW_VERSION);
}

/**
 * Prints one CLI error followed by help.
 * @param name Program executable name.
 * @param message Error text.
 * @return Process exit status.
 */
static int hnsw_fail_usage(const char *name, const char *message) {
    fprintf(stderr, "Error: %s\n\n", message);
    hnsw_help(name);
    return 1;
}

/**
 * Parses one signed integer.
 * @param text Source text.
 * @param out Destination integer pointer.
 * @return 1 on success, or 0 on failure.
 */
static int hnsw_parse_int(const char *text, int *out) {
    char *end;
    long value;

    if (text == NULL || out == NULL) {
        return 0;
    }

    errno = 0;
    value = strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }

    if (value < -2147483647L - 1L || value > 2147483647L) {
        return 0;
    }

    *out = (int)value;
    return 1;
}

/**
 * Parses one floating-point value.
 * @param text Source text.
 * @param out Destination value pointer.
 * @return 1 on success, or 0 on failure.
 */
static int hnsw_parse_float(const char *text, float *out) {
    char *end;
    float value;

    if (text == NULL || out == NULL) {
        return 0;
    }

    errno = 0;
    value = strtof(text, &end);

    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }

    *out = value;
    return 1;
}

/**
 * Parses one delimited vector into the provided buffer.
 * @param text Source vector text.
 * @param dimension Expected vector dimension.
 * @param out Destination vector buffer.
 * @return 1 on success, or 0 on failure.
 */
static int hnsw_parse_vector(const char *text, size_t dimension, float *out) {
    char buffer[HNSW_QUERY_CAP];
    char *token;
    char *state;
    size_t count;

    if (text == NULL || out == NULL || dimension == 0) {
        return 0;
    }

    if (strlen(text) >= sizeof(buffer)) {
        return 0;
    }

    memcpy(buffer, text, strlen(text) + 1);

    for (count = 0; buffer[count] != '\0'; count++) {
        if (buffer[count] == ',') {
            buffer[count] = ' ';
        }
    }

    count = 0;
    token = strtok_r(buffer, " \t\r\n", &state);

    while (token != NULL) {
        if (count >= dimension) {
            return 0;
        }

        if (!hnsw_parse_float(token, &out[count])) {
            return 0;
        }

        count++;
        token = strtok_r(NULL, " \t\r\n", &state);
    }

    return count == dimension;
}

/**
 * Loads one dataset file into the vector index.
 * @param hnsw Index pointer.
 * @param path Dataset file path.
 * @return Status code.
 */
static int hnsw_load_file(kc_hnsw_t *hnsw, const char *path) {
    FILE *file;
    char line[HNSW_LINE_CAP];
    float *values;
    int rc;

    if (hnsw == NULL || path == NULL) {
        return KC_HNSW_EINVAL;
    }

    file = fopen(path, "r");
    if (file == NULL) {
        return KC_HNSW_ESTATE;
    }

    values = (float *)malloc(kc_hnsw_dimension (hnsw) * sizeof(*values));
    if (values == NULL) {
        fclose(file);
        return KC_HNSW_ENOMEM;
    }

    rc = KC_HNSW_OK;

    while (fgets(line, sizeof(line), file) != NULL) {
        char *token;
        char *state;
        char *id;
        size_t index;

        token = strtok_r(line, " \t\r\n", &state);
        if (token == NULL || token[0] == '#') {
            continue;
        }

        id = token;
        index = 0;

        token = strtok_r(NULL, " \t\r\n", &state);
        while (token != NULL && index < kc_hnsw_dimension (hnsw)) {
            if (!hnsw_parse_float(token, &values[index])) {
                rc = KC_HNSW_EINVAL;
                break;
            }

            index++;
            token = strtok_r(NULL, " \t\r\n", &state);
        }

        if (rc != KC_HNSW_OK) {
            break;
        }

        if (index != kc_hnsw_dimension (hnsw) || token != NULL) {
            rc = KC_HNSW_EINVAL;
            break;
        }

        rc = kc_hnsw_add(hnsw, id, values);
        if (rc != KC_HNSW_OK) {
            break;
        }
    }

    free(values);
    fclose(file);
    return rc;
}

/**
 * Standalone entry point.
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument vector.
 * @return Process exit status.
 */
int main(int argc, char **argv) {
    kc_hnsw_t *hnsw;
    kc_hnsw_result_t results[HNSW_RESULT_CAP];
    float query[HNSW_QUERY_CAP];
    const char *dataset_path;
    const char *query_text;
    const char *metric_name;
    int dimension;
    int metric;
    int limit;
    int rc;
    int written;
    int i;

    dataset_path = NULL;
    query_text = NULL;
    metric_name = "cosine";
    dimension = 0;
    limit = 5;
    double threshold = -1e18; /* Default to allow everything for cosine/inner */

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            hnsw_help(argv[0]);
            return 0;
        }

        if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
            hnsw_version();
            return 0;
        }

        if (strcmp(argv[i], "--dim") == 0 || strcmp(argv[i], "-d") == 0) {
            if (i + 1 >= argc || !hnsw_parse_int(argv[i + 1], &dimension)) {
                return hnsw_fail_usage(argv[0], "Invalid value for --dim.");
            }

            i++;
            continue;
        }

        if (strcmp(argv[i], "--input") == 0 || strcmp(argv[i], "-i") == 0) {
            if (i + 1 >= argc) {
                return hnsw_fail_usage(argv[0], "Missing value for --input.");
            }

            dataset_path = argv[i + 1];
            i++;
            continue;
        }

        if (strcmp(argv[i], "--query") == 0 || strcmp(argv[i], "-q") == 0) {
            if (i + 1 >= argc) {
                return hnsw_fail_usage(argv[0], "Missing value for --query.");
            }

            query_text = argv[i + 1];
            i++;
            continue;
        }

        if (strcmp(argv[i], "--top") == 0 || strcmp(argv[i], "-k") == 0) {
            if (i + 1 >= argc || !hnsw_parse_int(argv[i + 1], &limit)) {
                return hnsw_fail_usage(argv[0], "Invalid value for --top.");
            }

            i++;
            continue;
        }

        if (strcmp(argv[i], "--metric") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 >= argc) {
                return hnsw_fail_usage(argv[0], "Missing value for --metric.");
            }

            metric_name = argv[i + 1];
            i++;
            continue;
        }

        if (strcmp(argv[i], "--threshold") == 0 || strcmp(argv[i], "-t") == 0) {
            float t;
            if (i + 1 >= argc || !hnsw_parse_float(argv[i + 1], &t)) {
                return hnsw_fail_usage(argv[0], "Invalid value for --threshold.");
            }

            threshold = (double)t;
            i++;
            continue;
        }

        return hnsw_fail_usage(argv[0], "Unknown argument.");
    }

    if (dimension <= 0) {
        return hnsw_fail_usage(argv[0], "Vector dimension must be greater than zero.");
    }

    if ((size_t)dimension > HNSW_QUERY_CAP) {
        return hnsw_fail_usage(argv[0], "Vector dimension is too large.");
    }

    if (dataset_path == NULL) {
        return hnsw_fail_usage(argv[0], "Missing --input dataset file.");
    }

    if (query_text == NULL) {
        return hnsw_fail_usage(argv[0], "Missing --query vector.");
    }

    if (limit < 1) {
        return hnsw_fail_usage(argv[0], "Top-K value must be greater than zero.");
    }

    if (limit > HNSW_RESULT_CAP) {
        limit = HNSW_RESULT_CAP;
    }

    metric = kc_hnsw_metric_from_string(metric_name);
    if (metric == 0) {
        return hnsw_fail_usage(argv[0], "Unknown metric name.");
    }

    if (!hnsw_parse_vector(query_text, (size_t)dimension, query)) {
        return hnsw_fail_usage(argv[0], "Query vector does not match the configured dimension.");
    }

    /* Adjust default threshold if not set by user */
    int threshold_set = 0;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--threshold") == 0 || strcmp(argv[i], "-t") == 0) {
            threshold_set = 1;
            break;
        }
    }
    if (!threshold_set) {
        if (metric == KC_HNSW_METRIC_L2) {
            threshold = 1e18;
        } else {
            threshold = -1e18;
        }
    }

    hnsw = kc_hnsw_open((size_t)dimension, metric);
    if (hnsw == NULL) {
        fprintf(stderr, "Error: initialization failed.\n");
        return 1;
    }

    rc = hnsw_load_file(hnsw, dataset_path);
    if (rc != KC_HNSW_OK) {
        fprintf(stderr, "Error: %s.\n", kc_hnsw_strerror(rc));
        kc_hnsw_close (hnsw);
        return 1;
    }

    rc = kc_hnsw_build (hnsw);
    if (rc != KC_HNSW_OK) {
        fprintf(stderr, "Error: index build failed (%s).\n", kc_hnsw_strerror(rc));
        kc_hnsw_close (hnsw);
        return 1;
    }

    written = kc_hnsw_search(hnsw, query, (size_t)limit, threshold, results);
    if (written < 0) {
        fprintf(stderr, "Error: %s.\n", kc_hnsw_strerror(written));
        kc_hnsw_close (hnsw);
        return 1;
    }

    for (i = 0; i < written; i++) {
        printf("%s: %.6f\n", results[i].id, results[i].score);
    }

    kc_hnsw_close (hnsw);
    return 0;
}
