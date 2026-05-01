# hnsw.c - HNSW Vector Search

A minimalist C library and CLI for fixed-dimension vector indexing with Approximate Nearest Neighbor search using a Hierarchical Navigable Small World (HNSW) graph.

---

## CLI

Run a nearest neighbor search over a vector dataset.

### Dataset format

Each line in the input file must contain:

```
<id> <v1> <v2> ... <vN>
```

Example (3D vectors):

```
item_1 1.0 0.0 0.0
item_2 0.0 1.0 0.0
item_3 0.0 0.0 1.0
```

---

### Examples

Basic search:

```bash
./bin/x86_64/linux/hnsw -d 3 -i vectors.txt -q "1 0 0"
```

Using a different metric and limiting results:

```bash
./bin/x86_64/linux/hnsw -d 3 -i vectors.txt -q "1 0 0" -m cosine -k 5
```

Applying a threshold:

```bash
./bin/x86_64/linux/hnsw -d 3 -i vectors.txt -q "1 0 0" -t 0.8
```

Pipe query vector through standard input:

```bash
echo "1 0 0" | ./bin/x86_64/linux/hnsw -d 3 -i vectors.txt
```

---

### Parameters

| Flag | Description |
| :--- | :--- |
| `-d` | Vector dimension |
| `-i` | Input dataset file |
| `-q` | Query vector |
| `-m` | Metric (`l2`, `cosine`, `ip`) |
| `-k` | Number of results |
| `-t` | Threshold filter |

---

### Output

Results are printed as:

```
<id>: <score>
```

---

## Metrics

Available metrics:

- `l2`: squared Euclidean distance
- `cosine`: cosine distance
- `ip`: inner product similarity

`l2` uses squared Euclidean distance:

```
d(a, b) = sum((a[i] - b[i])^2)
```

Note: no square root is applied. Rankings are identical to Euclidean distance.

---

## Public API

```c
#include "hnsw.h"

kc_hnsw_t *hnsw = kc_hnsw_open(dimension, KC_HNSW_METRIC_COSINE);
kc_hnsw_add(hnsw, "id_1", values);
kc_hnsw_build(hnsw);
kc_hnsw_search(hnsw, query, limit, threshold, results);
kc_hnsw_close(hnsw);
```

---

## Lifecycle

- `kc_hnsw_open()` allocates a new index.
- `kc_hnsw_add()` inserts vectors.
- `kc_hnsw_build()` constructs the HNSW graph.
- `kc_hnsw_search()` queries the index.
- `kc_hnsw_close()` releases all resources.

---

## Build

```bash
make all
make x86_64/linux
make x86_64/windows
make i686/linux
make i686/windows
make aarch64/linux
make aarch64/android
make armv7/linux
make armv7/android
make armv7hf/linux
make riscv64/linux
make powerpc64le/linux
make mips/linux
make mipsel/linux
make mips64el/linux
make s390x/linux
make loongarch64/linux
make clean
```

Artifacts are generated under:

```
bin/{arch}/{platform}/
```

---

**Author:** KaisarCode

**Email:** <kaisar@kaisarcode.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
