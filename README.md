# hnsw.c — HNSW Vector Search

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
- `libhnsw.a` — static library
- `libhnsw.so` / `libhnsw.dll` — shared library
- `hnsw` / `hnsw.exe` — CLI executable

## Usage

```bash
./bin/x86_64/linux/hnsw -d 3 -i vectors.txt -q "1 0 0"
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

- `kc_hnsw_open()` — allocates and returns an index owned by the caller.
- `kc_hnsw_close()` — releases the index. Must not be called while any other thread holds the index.
- `kc_hnsw_add()` and `kc_hnsw_build()` acquire an exclusive write lock. No other operation may run concurrently with them.
- `kc_hnsw_search()` acquires a shared read lock. Multiple threads may search the same index concurrently after `kc_hnsw_build()` completes.

---

**Author:** KaisarCode

**Email:** <kaisarcode@gmail.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
