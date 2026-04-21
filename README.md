# hnsw - HNSW Vector Search

`hnsw` is a small C library and CLI for fixed-dimension vector indexing with Approximate Nearest Neighbor search using a Hierarchical Navigable Small World (HNSW) graph.

## What It Does

For a collection of vectors, `hnsw`:

1. stores vectors of one fixed dimension
2. associates each vector with a user-defined identifier
3. constructs a hierarchical graph index for efficient navigation
4. returns the top-K closest closest matches for a query vector

The library uses HNSW to achieve sublinear search complexity, making it suitable for large-scale datasets where linear scans are impractical.

## Public API

```c
typedef struct kc_hnsw kc_hnsw_t;

typedef struct {
    const char *id;
    double score;
} kc_hnsw_result_t;

kc_hnsw_t *kc_hnsw_open(size_t dimension, int metric);
void kc_hnsw_close(kc_hnsw_t *hnsw.;

int kc_hnsw_reserve(kc_hnsw_t *hnsw, size_t capacity);
int kc_hnsw_add(kc_hnsw_t *hnsw, const char *id, const float *values);
int kc_hnsw_build(kc_hnsw_t *hnsw.;
int kc_hnsw_search(
    const kc_hnsw_t *hnsw,
    const float *query,
    size_t limit,
    double threshold,
    kc_hnsw_result_t *out
);

size_t kc_hnsw_dimension(const kc_hnsw_t *hnsw.;
int kc_hnsw_metric(const kc_hnsw_t *hnsw.;
size_t kc_hnsw_count(const kc_hnsw_t *hnsw.;

int kc_hnsw_metric_from_string(const char *name);
const char *kc_hnsw_metric_to_string(int metric);
const char *kc_hnsw_strerror(int rc);
```

### Search Thresholds

- `cosine` and `inner`: `score >= threshold`
- `l2`: `distance <= threshold`

## CLI Usage

```bash
hnsw -d 3 -i vectors.txt -q "1 0 0"
```

Dataset format:

```text
id_1 1 0 0
id_2 0 1 0
id_3 0.8 0.1 0
```

Options:

- `--dim`, `-d <n>`
- `--input`, `-i <path>`
- `--query`, `-q <values>`
- `--top`, `-k <n>`
- `--threshold`, `-t <n>`
- `--metric`, `-m <name>`
- `--help`, `-h`
- `--version`, `-v`

## Build

POSIX:

```bash
cc -O3 -std=c99 libhnsw.c hnsw.c -lm -o hnsw
./test.sh
```

Windows:

```bash
cl /O2 /std:c11 /TC libhnsw.c hnsw.c
```

The source code is written to stay portable across Windows, macOS, iOS,
Linux, and Android. Final compiler flags and output names depend on the toolchain.

## Library Example

```c
#include <stdio.h>
#include "hnsw.h"

int main(void) {
    kc_hnsw_result_t results[2];
    float a[3] = {1.0f, 0.0f, 0.0f};
    float b[3] = {0.0f, 1.0f, 0.0f};
    float q[3] = {0.9f, 0.1f, 0.0f};
    kc_hnsw_t *hnsw.
    int written;
    int i;

    hnsw = kc_hnsw_open(3, KC_HNSW_METRIC_COSINE);
    if (!hnsw. {
        return 1;
    }

    kc_hnsw_add(hnsw, "a", a);
    kc_hnsw_add(hnsw, "b", b);
    
    /* Construct the graph index */
    kc_hnsw_build(hnsw.;

    /* Search with a very low threshold to allow any match */
    written = kc_hnsw_search(hnsw, q, 2, -1.0, results);

    for (i = 0; i < written; i++) {
        printf("%s %.4f\n", results[i].id, results[i].score);
    }

    kc_hnsw_close(hnsw.;
    return 0;
}
```

---

**Author:** KaisarCode

**Email:** <kaisar@kaisarcode.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
