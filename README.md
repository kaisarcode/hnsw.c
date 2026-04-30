# hnsw.c - HNSW Vector Search

A minimalist C library for fixed-dimension vector indexing with Approximate Nearest Neighbor search using a Hierarchical Navigable Small World (HNSW) graph.

## File Layout

```
hnsw.c/
├── src/
│   ├── hnsw.c         CLI entry point (main)
│   ├── libhnsw.c      Core library implementation
│   └── hnsw.h         Public API header
├── bin/               Compiled artifacts (committed, Git LFS)
│   ├── x86_64/{linux,windows}
│   ├── i686/{linux,windows}
│   ├── aarch64/{linux,android}
│   ├── armv7/{linux,android}
│   ├── armv7hf/linux
│   ├── riscv64/linux
│   ├── powerpc64le/linux
│   ├── mips/linux  mipsel/linux  mips64el/linux
│   ├── s390x/linux
│   └── loongarch64/linux
├── CMakeLists.txt
├── Makefile
├── test.sh
└── README.md
```

## Build

```bash
make all              # all 16 targets
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

Each target produces under `bin/{arch}/{platform}/`:
- `libhnsw.a` - static library
- `libhnsw.so` / `libhnsw.dll` / `libhnsw.dll.a` - shared library and import lib
- `hnsw` / `hnsw.exe` - CLI executable

## CLI

```bash
./bin/x86_64/linux/hnsw -d 3 -i vectors.txt -q "1 0 0"
./bin/x86_64/linux/hnsw -d 3 -i vectors.txt -q "1 0 0" -m cosine -k 5 -t 0.8
```

### Arguments

- `-d <int>`
    Vector dimension. Must match the number of components in each input vector.

- `-i <file>`
    Input dataset file (see Dataset Format section).

- `-q "<values>"`
    Query vector as space-separated numbers. Must match dimension `-d`.

- `-m <metric>` (optional)
    Distance metric. Default: `l2`.

    Supported values:
    - `l2`: squared Euclidean distance
    - `cosine`: cosine distance
    - `ip`: inner product similarity

- `-k <int>` (optional)
    Number of nearest neighbors to return. Default: implementation-defined.

- `-t <float>` (optional)
    Threshold for filtering results. Interpretation depends on metric:
    - `l2`: maximum distance
    - `cosine`: maximum distance
    - `ip`: minimum similarity

### Output

The CLI prints results as:

    <id>: <score>

Where `<score>` is the distance or similarity depending on the selected metric.

## Dataset Format

The input file passed with `-i` must contain one vector per line.

Each line format is:

    <id> <v1> <v2> ... <vN>

Where:

- `<id>` is a non-empty identifier without whitespace.
- `<v1> ... <vN>` are numeric vector components.
- `N` must match the dimension passed with `-d`.

Example for `-d 3`:

    item_1 1.0 0.0 0.0
    item_2 0.0 1.0 0.0
    item_3 0.0 0.0 1.0

Then this command queries the same 3-dimensional space:

    ./bin/x86_64/linux/hnsw -d 3 -i vectors.txt -q "1 0 0"

## Metrics

Available metrics:
- `l2`: squared Euclidean distance
- `cosine`: cosine distance
- `ip`: inner product similarity

`l2` uses squared Euclidean distance:

    d(a, b) = sum((a[i] - b[i])^2)

Note: no square root is applied. Rankings are identical to Euclidean distance.

### Search Tuning

The search budget (`ef`) can be adjusted via the `HNSW_EF_SEARCH` environment variable (default: 64). Higher values increase accuracy at the cost of performance.

```bash
env HNSW_EF_SEARCH=128 ./bin/x86_64/linux/hnsw ...
```

## Public API

```c
#include "hnsw.h"

kc_hnsw_t *hnsw = kc_hnsw_open(dimension, KC_HNSW_METRIC_COSINE);
kc_hnsw_add(hnsw, "id_1", values);
kc_hnsw_build(hnsw);
kc_hnsw_search(hnsw, query, limit, threshold, results);
kc_hnsw_close(hnsw);
```

## Lifecycle

- `kc_hnsw_open()` - allocates and returns an index owned by the caller.
- `kc_hnsw_close()` - releases the index. Must not be called while any other thread holds the index.
- `kc_hnsw_add()` and `kc_hnsw_build()` acquire an exclusive write lock. No other operation may run concurrently with them.
- `kc_hnsw_search()` acquires a shared read lock. Multiple threads may search the same index concurrently after `kc_hnsw_build()` completes.

---

**Author:** KaisarCode

**Email:** <kaisarcode@gmail.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
