/**
 * libhnsw.c - HNSW Vector Search
 * Summary: HNSW-based approximate nearest neighbor search implementation.
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "hnsw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define KC_HNSW_HNSW_M 16
#define KC_HNSW_HNSW_EF_CONSTRUCTION 64
#define KC_HNSW_HNSW_EF_SEARCH 64
#define KC_HNSW_HNSW_MULT 0.42

typedef struct {
    size_t target_idx;
} kc_hnsw_edge_t;

typedef struct {
    kc_hnsw_edge_t *edges;
    size_t count;
    size_t capacity;
} kc_hnsw_neighbor_list_t;

typedef struct {
    char *id;
    float *values;
    double norm;
    int level;
    kc_hnsw_neighbor_list_t *neighbors;
} kc_hnsw_item_t;

struct kc_hnsw {
    size_t dimension;
    int metric;
    kc_hnsw_item_t *items;
    size_t count;
    size_t capacity;

    int max_level;
    size_t entry_point_idx;
    int entry_point_set;

    int M;
    int ef_construction;
    int ef_search;
};

/* Priority Queue / Heap for HNSW search */
typedef struct {
    size_t idx;
    double score;
} kc_hnsw_node_score_t;

typedef struct {
    kc_hnsw_node_score_t *data;
    size_t size;
    size_t capacity;
    int metric;
    int is_max_heap; /* 1 for max-heap, 0 for min-heap */
} kc_hnsw_heap_t;

static char *kc_hnsw_strdup(const char *text);
static int kc_hnsw_metric_valid(int metric);
static double kc_hnsw_vector_norm(const float *values, size_t dimension);
static double kc_hnsw_inner_product(const float *left, const float *right, size_t dimension);
static double kc_hnsw_dist(const kc_hnsw_t *hnsw, const float *v1, double n1, const float *v2, double n2);

static int kc_hnsw_random_level(void);
static kc_hnsw_heap_t *kc_hnsw_heap_create(size_t capacity, int metric, int is_max_heap);
static void kc_hnsw_heap_destroy(kc_hnsw_heap_t *heap);
static void kc_hnsw_heap_push(kc_hnsw_heap_t *heap, size_t idx, double score);
static kc_hnsw_node_score_t kc_hnsw_heap_pop(kc_hnsw_heap_t *heap);
static int kc_hnsw_heap_is_better(int metric, double s1, double s2, int is_max_heap);

static void kc_hnsw_search_level(const kc_hnsw_t *hnsw, const float *query, double query_norm, size_t entry_idx, int level, int ef, kc_hnsw_heap_t *results);
static void kc_hnsw_add_edge(kc_hnsw_t *hnsw, size_t src_idx, size_t dst_idx, int level);
static void kc_hnsw_neighbor_list_init(kc_hnsw_neighbor_list_t *list);
static void kc_hnsw_neighbor_list_free(kc_hnsw_neighbor_list_t *list);

/**
 * Creates one vector index instance.
 * @param dimension Fixed vector dimension for all entries.
 * @param metric Configured similarity metric.
 * @return Index pointer or NULL on allocation failure.
 */
kc_hnsw_t *kc_hnsw_open(size_t dimension, int metric) {
    kc_hnsw_t *hnsw;

    if (dimension == 0 || !kc_hnsw_metric_valid(metric)) {
        return NULL;
    }

    hnsw = (kc_hnsw_t *)calloc(1, sizeof(*hnsw));
    if (hnsw == NULL) {
        return NULL;
    }

    hnsw->dimension = dimension;
    hnsw->metric = metric;
    hnsw->max_level = -1;
    hnsw->entry_point_set = 0;
    hnsw->M = KC_HNSW_HNSW_M;
    hnsw->ef_construction = KC_HNSW_HNSW_EF_CONSTRUCTION;
    hnsw->ef_search = KC_HNSW_HNSW_EF_SEARCH;

    srand((unsigned int)time(NULL));

    return hnsw;
}

/**
 * Releases one vector index instance.
 * @param hnsw Index pointer.
 * @return No return value.
 */
void kc_hnsw_close(kc_hnsw_t *hnsw) {
    size_t i;
    int j;

    if (hnsw == NULL) {
        return;
    }

    for (i = 0; i < hnsw->count; i++) {
        free(hnsw->items[i].id);
        free(hnsw->items[i].values);
        if (hnsw->items[i].neighbors) {
            for (j = 0; j <= hnsw->items[i].level; j++) {
                kc_hnsw_neighbor_list_free(&hnsw->items[i].neighbors[j]);
            }
            free(hnsw->items[i].neighbors);
        }
    }

    free(hnsw->items);
    free (hnsw);
}

/**
 * Reserves capacity for a target number of vectors.
 * @param hnsw Index pointer.
 * @param capacity Target vector capacity.
 * @return Status code.
 */
int kc_hnsw_reserve(kc_hnsw_t *hnsw, size_t capacity) {
    kc_hnsw_item_t *items;

    if (hnsw == NULL) {
        return KC_HNSW_EINVAL;
    }

    if (capacity <= hnsw->capacity) {
        return KC_HNSW_OK;
    }

    items = (kc_hnsw_item_t *)realloc(hnsw->items, capacity * sizeof(*items));
    if (items == NULL) {
        return KC_HNSW_ENOMEM;
    }

    hnsw->items = items;
    hnsw->capacity = capacity;
    return KC_HNSW_OK;
}

/**
 * Inserts one vector and its identifier into the index.
 * @param hnsw Index pointer.
 * @param id User-defined identifier string.
 * @param values Vector values with the configured dimension.
 * @return Status code.
 */
int kc_hnsw_add(kc_hnsw_t *hnsw, const char *id, const float *values) {
    float *copy;
    char *id_copy;
    size_t next_capacity;

    if (hnsw == NULL || id == NULL || id[0] == '\0' || values == NULL) {
        return KC_HNSW_EINVAL;
    }

    if (hnsw->count == hnsw->capacity) {
        next_capacity = hnsw->capacity == 0 ? 8 : hnsw->capacity * 2;
        if (kc_hnsw_reserve(hnsw, next_capacity) != KC_HNSW_OK) {
            return KC_HNSW_ENOMEM;
        }
    }

    id_copy = kc_hnsw_strdup(id);
    if (id_copy == NULL) {
        return KC_HNSW_ENOMEM;
    }

    copy = (float *)malloc(hnsw->dimension * sizeof(*copy));
    if (copy == NULL) {
        free(id_copy);
        return KC_HNSW_ENOMEM;
    }

    memcpy(copy, values, hnsw->dimension * sizeof(*copy));

    hnsw->items[hnsw->count].id = id_copy;
    hnsw->items[hnsw->count].values = copy;
    hnsw->items[hnsw->count].norm = kc_hnsw_vector_norm(copy, hnsw->dimension);
    hnsw->items[hnsw->count].level = -1;
    hnsw->items[hnsw->count].neighbors = NULL;
    hnsw->count++;

    return KC_HNSW_OK;
}

/**
 * HNSW Build Phase.
 * @param hnsw Index pointer.
 * @return Status code.
 */
int kc_hnsw_build(kc_hnsw_t *hnsw) {
    if (hnsw == NULL) return KC_HNSW_EINVAL;
    if (hnsw->count == 0) return KC_HNSW_OK;

    /* Reset graph if already built */
    for (size_t i = 0; i < hnsw->count; i++) {
        if (hnsw->items[i].neighbors) {
            for (int j = 0; j <= hnsw->items[i].level; j++) {
                kc_hnsw_neighbor_list_free(&hnsw->items[i].neighbors[j]);
            }
            free(hnsw->items[i].neighbors);
            hnsw->items[i].neighbors = NULL;
        }
    }
    hnsw->max_level = -1;
    hnsw->entry_point_set = 0;

    /* Shuffle for better graph balance */
    for (size_t i = 0; i < hnsw->count; i++) {
        size_t j = i + rand() % (hnsw->count - i);
        kc_hnsw_item_t temp = hnsw->items[i];
        hnsw->items[i] = hnsw->items[j];
        hnsw->items[j] = temp;
    }

    for (size_t i = 0; i < hnsw->count; i++) {
        int level = kc_hnsw_random_level();
        hnsw->items[i].level = level;
        hnsw->items[i].neighbors = (kc_hnsw_neighbor_list_t *)calloc(level + 1, sizeof(kc_hnsw_neighbor_list_t));
        for (int j = 0; j <= level; j++) {
            kc_hnsw_neighbor_list_init(&hnsw->items[i].neighbors[j]);
        }

        if (!hnsw->entry_point_set) {
            hnsw->entry_point_idx = i;
            hnsw->max_level = level;
            hnsw->entry_point_set = 1;
            continue;
        }

        size_t curr_idx = hnsw->entry_point_idx;
        double curr_dist = kc_hnsw_dist(hnsw, hnsw->items[i].values, hnsw->items[i].norm, hnsw->items[curr_idx].values, hnsw->items[curr_idx].norm);

        /* 1. Greedy descent to level */
        for (int l = hnsw->max_level; l > level; l--) {
            int changed = 1;
            while (changed) {
                changed = 0;
                kc_hnsw_neighbor_list_t *neighbors = &hnsw->items[curr_idx].neighbors[l];
                for (size_t n = 0; n < neighbors->count; n++) {
                    size_t neighbor_idx = neighbors->edges[n].target_idx;
                    double d = kc_hnsw_dist(hnsw, hnsw->items[i].values, hnsw->items[i].norm, hnsw->items[neighbor_idx].values, hnsw->items[neighbor_idx].norm);
                    if (kc_hnsw_heap_is_better(hnsw->metric, d, curr_dist, 0)) {
                        curr_dist = d;
                        curr_idx = neighbor_idx;
                        changed = 1;
                    }
                }
            }
        }

        /* 2. Insert into levels */
        for (int l = (level < hnsw->max_level ? level : hnsw->max_level); l >= 0; l--) {
            kc_hnsw_heap_t *candidates = kc_hnsw_heap_create(hnsw->ef_construction, hnsw->metric, 0);
            kc_hnsw_search_level(hnsw, hnsw->items[i].values, hnsw->items[i].norm, curr_idx, l, hnsw->ef_construction, candidates);
            
            /* Connect to top M neighbors */
            int connected = 0;
            while (candidates->size > 0 && connected < hnsw->M) {
                kc_hnsw_node_score_t best = kc_hnsw_heap_pop(candidates);
                kc_hnsw_add_edge(hnsw, i, best.idx, l);
                kc_hnsw_add_edge(hnsw, best.idx, i, l);
                
                if (connected == 0) {
                    curr_idx = best.idx;
                    curr_dist = best.score;
                }
                connected++;
            }
            kc_hnsw_heap_destroy(candidates);
        }

        if (level > hnsw->max_level) {
            hnsw->max_level = level;
            hnsw->entry_point_idx = i;
        }
    }

    return KC_HNSW_OK;
}

/**
 * HNSW Search logic.
 * @param hnsw Index pointer.
 * @param query Query vector.
 * @param limit Maximum number of results to write.
 * @param threshold Minimum score or maximum distance to accept.
 * @param out Caller-provided output buffer.
 * @return Number of results written, or a negative status code on failure.
 */
int kc_hnsw_search(const kc_hnsw_t *hnsw, const float *query, size_t limit, double threshold, kc_hnsw_result_t *out) {
    if (hnsw == NULL || query == NULL || (limit > 0 && out == NULL)) return KC_HNSW_EINVAL;
    if (limit == 0 || hnsw->count == 0) return 0;
    if (hnsw->count > 0 && !hnsw->entry_point_set) return KC_HNSW_ESTATE;

    double q_norm = kc_hnsw_vector_norm(query, hnsw->dimension);
    size_t curr_idx = hnsw->entry_point_idx;
    double curr_dist = kc_hnsw_dist(hnsw, query, q_norm, hnsw->items[curr_idx].values, hnsw->items[curr_idx].norm);

    /* 1. Greedy descent to level 0 */
    for (int l = hnsw->max_level; l > 0; l--) {
        int changed = 1;
        while (changed) {
            changed = 0;
            kc_hnsw_neighbor_list_t *neighbors = &hnsw->items[curr_idx].neighbors[l];
            for (size_t n = 0; n < neighbors->count; n++) {
                size_t neighbor_idx = neighbors->edges[n].target_idx;
                double d = kc_hnsw_dist(hnsw, query, q_norm, hnsw->items[neighbor_idx].values, hnsw->items[neighbor_idx].norm);
                if (kc_hnsw_heap_is_better(hnsw->metric, d, curr_dist, 0)) {
                    curr_dist = d;
                    curr_idx = neighbor_idx;
                    changed = 1;
                }
            }
        }
    }

    /* 2. Best-first search at level 0 */
    kc_hnsw_heap_t *top_k = kc_hnsw_heap_create(hnsw->ef_search, hnsw->metric, 1);
    kc_hnsw_search_level(hnsw, query, q_norm, curr_idx, 0, hnsw->ef_search, top_k);

    /* 3. Filter and return */
    size_t written = 0;
    /* We need to extract from max-heap and reverse to get best first, OR just extract all and sort. */
    /* Actually our heap is a max-heap of size K, so popping will give us worst to best. */
    kc_hnsw_node_score_t *results = (kc_hnsw_node_score_t *)malloc(top_k->size * sizeof(kc_hnsw_node_score_t));
    size_t res_count = top_k->size;
    for (int i = (int)res_count - 1; i >= 0; i--) {
        results[i] = kc_hnsw_heap_pop(top_k);
    }

    for (size_t i = 0; i < res_count && written < limit; i++) {
        int match = 0;
        if (hnsw->metric == KC_HNSW_METRIC_L2) {
            if (results[i].score <= threshold) match = 1;
        } else {
            if (results[i].score >= threshold) match = 1;
        }

        if (match) {
            out[written].id = hnsw->items[results[i].idx].id;
            out[written].score = results[i].score;
            written++;
        }
    }

    free(results);
    kc_hnsw_heap_destroy(top_k);
    return (int)written;
}

/* Internal implementation */

/**
 * Performs a search at a specific HNSW level.
 * @param hnsw Index pointer.
 * @param query Query vector.
 * @param query_norm Precomputed query norm.
 * @param entry_idx Starting node index.
 * @param level Graph level to search.
 * @param ef Search budget (ef).
 * @param results Output heap to store found nodes.
 * @return No return value.
 */
static void kc_hnsw_search_level(const kc_hnsw_t *hnsw, const float *query, double query_norm, size_t entry_idx, int level, int ef, kc_hnsw_heap_t *results) {
    kc_hnsw_heap_t *candidates = kc_hnsw_heap_create(ef * 2, hnsw->metric, 0); /* min-heap */
    char *visited = (char *)calloc(hnsw->count, 1);

    double d = kc_hnsw_dist(hnsw, query, query_norm, hnsw->items[entry_idx].values, hnsw->items[entry_idx].norm);
    kc_hnsw_heap_push(candidates, entry_idx, d);
    kc_hnsw_heap_push(results, entry_idx, d);
    visited[entry_idx] = 1;

    while (candidates->size > 0) {
        kc_hnsw_node_score_t c = kc_hnsw_heap_pop(candidates);
        
        /* results is a max-heap of size ef (worst candidate at top) */
        /* If current candidate is worse than the worst result, we can stop if it's greedy enough, */
        /* but here we follow ef. */
        kc_hnsw_node_score_t worst_res = results->data[0];
        if (kc_hnsw_heap_is_better(hnsw->metric, c.score, worst_res.score, 1)) {
            /* In HNSW, if c is farther than the furthest in results, we might skip, but let's be thorough */
            /* Actually, if c.score is worse than worst_res.score, we don't necessarily stop yet because we explore neighbors. */
        }

        kc_hnsw_neighbor_list_t *neighbors = &hnsw->items[c.idx].neighbors[level];
        for (size_t n = 0; n < neighbors->count; n++) {
            size_t v_idx = neighbors->edges[n].target_idx;
            if (!visited[v_idx]) {
                visited[v_idx] = 1;
                double v_dist = kc_hnsw_dist(hnsw, query, query_norm, hnsw->items[v_idx].values, hnsw->items[v_idx].norm);
                
                worst_res = results->data[0];
                if (results->size < (size_t)ef || kc_hnsw_heap_is_better(hnsw->metric, v_dist, worst_res.score, 0)) {
                    kc_hnsw_heap_push(candidates, v_idx, v_dist);
                    kc_hnsw_heap_push(results, v_idx, v_dist);
                    if (results->size > (size_t)ef) {
                        kc_hnsw_heap_pop(results);
                    }
                }
            }
        }
    }

    free(visited);
    kc_hnsw_heap_destroy(candidates);
}

/**
 * Adds a directed edge between two nodes in the graph.
 * @param hnsw Index pointer.
 * @param src_idx Source node index.
 * @param dst_idx Destination node index.
 * @param level Graph level.
 * @return No return value.
 */
static void kc_hnsw_add_edge(kc_hnsw_t *hnsw, size_t src_idx, size_t dst_idx, int level) {
    kc_hnsw_neighbor_list_t *list = &hnsw->items[src_idx].neighbors[level];
    
    /* Check if already exists */
    for (size_t i = 0; i < list->count; i++) {
        if (list->edges[i].target_idx == dst_idx) return;
    }

    if (list->count == list->capacity) {
        size_t next_cap = list->capacity == 0 ? 4 : list->capacity * 2;
        list->edges = (kc_hnsw_edge_t *)realloc(list->edges, next_cap * sizeof(kc_hnsw_edge_t));
        list->capacity = next_cap;
    }
    list->edges[list->count++].target_idx = dst_idx;

    /* Pruning: if count > M_max, keep only top M_max. */
    size_t M_max = (level == 0) ? (size_t)hnsw->M * 2 : (size_t)hnsw->M;
    if (list->count > M_max) {
        int worst_idx = -1;
        double worst_score = 0;

        for (size_t i = 0; i < list->count; i++) {
            size_t n_idx = list->edges[i].target_idx;
            double s = kc_hnsw_dist(hnsw, hnsw->items[src_idx].values, hnsw->items[src_idx].norm,
                    hnsw->items[n_idx].values, hnsw->items[n_idx].norm);
            if (worst_idx == -1 || kc_hnsw_heap_is_better(hnsw->metric, s, worst_score, 1)) {
                worst_score = s;
                worst_idx = (int)i;
            }
        }

        if (worst_idx != -1) {
            list->edges[worst_idx] = list->edges[list->count - 1];
            list->count--;
        }
    }
}

/**
 * Generates a random level for a new node.
 * @return Randomized level index.
 */
static int kc_hnsw_random_level(void) {
    double r = (double)rand() / RAND_MAX;
    int level = 0;
    while (r < KC_HNSW_HNSW_MULT && level < 16) {
        level++;
        r = (double)rand() / RAND_MAX;
    }
    return level;
}

/**
 * Initializes a neighbor list structure.
 * @param list List pointer.
 * @return No return value.
 */
static void kc_hnsw_neighbor_list_init(kc_hnsw_neighbor_list_t *list) {
    list->edges = NULL;
    list->count = 0;
    list->capacity = 0;
}

/**
 * Releases a neighbor list structure.
 * @param list List pointer.
 * @return No return value.
 */
static void kc_hnsw_neighbor_list_free(kc_hnsw_neighbor_list_t *list) {
    free(list->edges);
}

/* Heap Implementation */

/**
 * Creates a new priority heap.
 * @param capacity Initial capacity.
 * @param metric Metric constant.
 * @param is_max_heap Flag for max-heap (1) or min-heap (0).
 * @return Heap pointer.
 */
static kc_hnsw_heap_t *kc_hnsw_heap_create(size_t capacity, int metric, int is_max_heap) {
    kc_hnsw_heap_t *heap = (kc_hnsw_heap_t *)malloc(sizeof(kc_hnsw_heap_t));
    heap->data = (kc_hnsw_node_score_t *)malloc(capacity * sizeof(kc_hnsw_node_score_t));
    heap->size = 0;
    heap->capacity = capacity;
    heap->metric = metric;
    heap->is_max_heap = is_max_heap;
    return heap;
}

/**
 * Releases a priority heap.
 * @param heap Heap pointer.
 * @return No return value.
 */
static void kc_hnsw_heap_destroy(kc_hnsw_heap_t *heap) {
    free(heap->data);
    free(heap);
}

/**
 * Compares two scores based on metric and heap type.
 * @param metric Metric constant.
 * @param s1 First score.
 * @param s2 Second score.
 * @param is_max_heap Flag for max-heap (1) or min-heap (0).
 * @return 1 if s1 is better than s2, or 0 otherwise.
 */
static int kc_hnsw_heap_is_better(int metric, double s1, double s2, int is_max_heap) {
    if (metric == KC_HNSW_METRIC_L2) {
        return is_max_heap ? (s1 > s2) : (s1 < s2);
    } else {
        return is_max_heap ? (s1 < s2) : (s1 > s2);
    }
}

/**
 * Pushes a new node into the heap.
 * @param heap Heap pointer.
 * @param idx Node index.
 * @param score Similarity score or distance.
 * @return No return value.
 */
static void kc_hnsw_heap_push(kc_hnsw_heap_t *heap, size_t idx, double score) {
    if (heap->size == heap->capacity) {
        heap->capacity *= 2;
        heap->data = (kc_hnsw_node_score_t *)realloc(heap->data, heap->capacity * sizeof(kc_hnsw_node_score_t));
    }
    size_t i = heap->size++;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!kc_hnsw_heap_is_better(heap->metric, score, heap->data[p].score, heap->is_max_heap)) break;
        heap->data[i] = heap->data[p];
        i = p;
    }
    heap->data[i].idx = idx;
    heap->data[i].score = score;
}

/**
 * Pops the best node from the heap.
 * @param heap Heap pointer.
 * @return Best node score structure.
 */
static kc_hnsw_node_score_t kc_hnsw_heap_pop(kc_hnsw_heap_t *heap) {
    kc_hnsw_node_score_t top = heap->data[0];
    kc_hnsw_node_score_t last = heap->data[--heap->size];
    size_t i = 0;
    while (i * 2 + 1 < heap->size) {
        size_t c = i * 2 + 1;
        if (c + 1 < heap->size && kc_hnsw_heap_is_better(heap->metric, heap->data[c+1].score, heap->data[c].score, heap->is_max_heap)) c++;
        if (!kc_hnsw_heap_is_better(heap->metric, heap->data[c].score, last.score, heap->is_max_heap)) break;
        heap->data[i] = heap->data[c];
        i = c;
    }
    heap->data[i] = last;
    return top;
}

/* Existing utility functions from libhnsw.c */

/**
 * Duplicates one string into heap memory.
 * @param text Source string.
 * @return Heap copy or NULL on allocation failure.
 */
static char *kc_hnsw_strdup(const char *text) {
    size_t size = strlen(text) + 1;
    char *copy = (char *)malloc(size);
    if (copy) memcpy(copy, text, size);
    return copy;
}

/**
 * Checks whether one metric constant is supported.
 * @param metric Metric constant.
 * @return 1 when valid, or 0 when invalid.
 */
static int kc_hnsw_metric_valid(int metric) {
    return (metric == KC_HNSW_METRIC_COSINE || metric == KC_HNSW_METRIC_INNER_PRODUCT || metric == KC_HNSW_METRIC_L2);
}

/**
 * Computes one vector Euclidean norm.
 * @param values Vector values.
 * @param dimension Vector dimension.
 * @return Non-negative norm.
 */
static double kc_hnsw_vector_norm(const float *values, size_t dimension) {
    double sum = 0.0;
    for (size_t i = 0; i < dimension; i++) sum += (double)values[i] * (double)values[i];
    return sqrt(sum);
}

/**
 * Computes one vector inner product.
 * @param left Left vector.
 * @param right Right vector.
 * @param dimension Vector dimension.
 * @return Dot-product value.
 */
static double kc_hnsw_inner_product(const float *left, const float *right, size_t dimension) {
    double sum = 0.0;
    for (size_t i = 0; i < dimension; i++) sum += (double)left[i] * (double)right[i];
    return sum;
}

/**
 * Computes the distance/similarity between two vectors.
 * @param hnsw Index pointer.
 * @param v1 First vector.
 * @param n1 Precomputed norm of v1.
 * @param v2 Second vector.
 * @param n2 Precomputed norm of v2.
 * @return Distance or similarity score.
 */
static double kc_hnsw_dist(const kc_hnsw_t *hnsw, const float *v1, double n1, const float *v2, double n2) {
    double ip = kc_hnsw_inner_product(v1, v2, hnsw->dimension);
    if (hnsw->metric == KC_HNSW_METRIC_INNER_PRODUCT) return ip;
    if (hnsw->metric == KC_HNSW_METRIC_COSINE) return (n1 == 0.0 || n2 == 0.0) ? 0.0 : ip / (n1 * n2);
    return n1 * n1 + n2 * n2 - 2.0 * ip;
}

/**
 * Returns the configured vector dimension.
 * @param hnsw Index pointer.
 * @return Dimension value, or 0 on invalid input.
 */
size_t kc_hnsw_dimension(const kc_hnsw_t *hnsw) { return hnsw ? hnsw->dimension : 0; }

/**
 * Returns the configured similarity metric.
 * @param hnsw Index pointer.
 * @return Metric constant, or 0 on invalid input.
 */
int kc_hnsw_metric(const kc_hnsw_t *hnsw) { return hnsw ? hnsw->metric : 0; }

/**
 * Returns the number of inserted vectors.
 * @param hnsw Index pointer.
 * @return Vector count, or 0 on invalid input.
 */
size_t kc_hnsw_count(const kc_hnsw_t *hnsw) { return hnsw ? hnsw->count : 0; }

/**
 * Resolves one metric name into a metric constant.
 * @param name Metric text name.
 * @return Metric constant, or 0 on invalid input.
 */
int kc_hnsw_metric_from_string(const char *name) {
    if (!name) return 0;
    if (strcmp(name, "cosine") == 0) return KC_HNSW_METRIC_COSINE;
    if (strcmp(name, "inner") == 0 || strcmp(name, "inner_product") == 0) return KC_HNSW_METRIC_INNER_PRODUCT;
    if (strcmp(name, "l2") == 0 || strcmp(name, "euclidean") == 0) return KC_HNSW_METRIC_L2;
    return 0;
}

/**
 * Resolves one metric constant into a metric name.
 * @param metric Metric constant.
 * @return Static metric name, or NULL on invalid input.
 */
const char *kc_hnsw_metric_to_string(int metric) {
    switch (metric) {
        case KC_HNSW_METRIC_COSINE: return "cosine";
        case KC_HNSW_METRIC_INNER_PRODUCT: return "inner";
        case KC_HNSW_METRIC_L2: return "l2";
        default: return NULL;
    }
}

/**
 * Resolves one status code into text.
 * @param rc Status code.
 * @return Static string.
 */
const char *kc_hnsw_strerror(int rc) {
    switch (rc) {
        case KC_HNSW_OK: return "ok";
        case KC_HNSW_EINVAL: return "invalid argument";
        case KC_HNSW_ENOMEM: return "out of memory";
        case KC_HNSW_ESTATE: return "invalid state";
        default: return "unknown error";
    }
}
