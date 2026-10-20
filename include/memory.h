#ifndef CENGINE_MEMORY_H
#define CENGINE_MEMORY_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "cengine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Cache-line alignment constant */
#define CE_CACHE_LINE_SIZE 64
#define CE_SIMD_ALIGNMENT 32  /* AVX2: 256-bit = 32 bytes */

/* ---- Aligned Allocation ---- */
void* ce_aligned_alloc(size_t alignment, size_t size);
void  ce_aligned_free(void* ptr);

/* ---- Memory Pool (slab allocator) ---- */
typedef struct ce_mem_pool ce_mem_pool_t;

ce_mem_pool_t* ce_mem_pool_create(size_t total_size);
void           ce_mem_pool_destroy(ce_mem_pool_t* pool);
void*          ce_mem_pool_alloc(ce_mem_pool_t* pool, size_t size);
void           ce_mem_pool_free(ce_mem_pool_t* pool, void* ptr);
void           ce_mem_pool_reset(ce_mem_pool_t* pool);

/* ---- Memory Tracker ---- */
typedef struct {
    int64_t current_bytes;
    int64_t peak_bytes;
    int64_t total_allocs;
    int64_t total_frees;
    int64_t active_allocs;       /* total_allocs - total_frees */
    int64_t alloc_bytes_rate;    /* bytes allocated in last interval */
    int64_t free_bytes_rate;     /* bytes freed in last interval */
    double  growth_rate;         /* bytes/sec change in current_bytes */
    double  fragmentation;       /* 1.0 - (useful / allocated) */
    int64_t tensor_count;        /* live tensors */
    int64_t tensor_total_bytes;  /* total tensor data bytes */
} ce_memory_stats_t;

ce_status_t          ce_tracker_init(void);
void                 ce_tracker_shutdown(void);
void                 ce_tracker_record_alloc(void* ptr, size_t size, const char* tag);
void                 ce_tracker_record_free(void* ptr);
void                 ce_tracker_update_tensor(int64_t count_delta, int64_t bytes_delta);
ce_memory_stats_t    ce_tracker_get_stats(void);
void                 ce_tracker_reset_rates(void);

/* Leak check: returns number of unfreed allocations */
int                  ce_tracker_check_leaks(void);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_MEMORY_H */
