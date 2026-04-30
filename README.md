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

The CLI expects a dataset file where each line starts with an identifier followed by the vector values (e.g., `id_1 0.1 0.2 0.3`). It outputs results as `id: score` pairs.

## Metrics

Available metrics:
- `l2`: squared Euclidean distance
- `cosine`: cosine distance
- `ip`: negative inner product

`l2` uses squared Euclidean distance:

    d(a, b) = sum((a[i] - b[i])^2)

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
