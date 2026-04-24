# hnsw - HNSW Vector Search

A minimalist C library for fixed-dimension vector indexing with Approximate Nearest Neighbor search using a Hierarchical Navigable Small World (HNSW) graph.

---

## Quick Start

### Build
Requires a C compiler and CMake 3.14+.
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
```
*The `hnsw` binary will be generated directly in the root directory.*

### Usage
```bash
./hnsw -d 3 -i vectors.txt -q "1 0 0"
```

---

## Features
- **Fast Navigation**: Sublinear search complexity using HNSW graph indexing.
- **Unified Build**: Single CMake workflow for all platforms.
- **Native Performance**: Optimized for CPU inference with SIMD support.
- **Multiple Metrics**: Supports Cosine, Inner Product, and L2 distance.

---

## Thread-Safety

hnsw uses a **build-then-query** model:

- Create the index, add all vectors, and call kc_hnsw_build() from one thread. During this build phase, the index is mutable and must not be accessed from other threads.
- After kc_hnsw_build() returns KC_HNSW_OK, the index becomes read-only for search. At that point, multiple threads may call kc_hnsw_search() concurrently on the same kc_hnsw_t.
- Each search thread must provide its own query buffer and its own result buffer.
- Do not add vectors, reserve capacity, rebuild, or close the index while searches are running.
- kc_hnsw_close() must be called only after all search threads have finished using the index.

---

## Public API
```c
#include "hnsw.h"

// Initialize HNSW index
kc_hnsw_t *hnsw = kc_hnsw_open(dimension, KC_HNSW_METRIC_COSINE);

// Add vectors
kc_hnsw_add(hnsw, "id_1", values);

// Build the index graph
kc_hnsw_build(hnsw);

// Search for nearest neighbors
kc_hnsw_search(hnsw, query, limit, threshold, results);

// Clean up
kc_hnsw_close(hnsw);
```

---

**Author:** KaisarCode

**Email:** <kaisar@kaisarcode.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
