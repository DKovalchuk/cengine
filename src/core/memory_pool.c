#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Slab size classes ---- */
#define SLAB_CLASSES 6

static const size_t slab_sizes[SLAB_CLASSES] = {
    64, 256, 1024, 4096, 16384, 65536
};

typedef struct slab_block {
    struct slab_block* next;
} slab_block_t;

typedef struct {
    slab_block_t* free_list;
    size_t block_size;
    int64_t alloc_count;
    int64_t free_count;
} slab_class_t;

struct ce_mem_pool {
    void* base;
    size_t total_size;
    size_t used;
    slab_class_t classes[SLAB_CLASSES];
};

/* Find best slab class for given size */
static int find_slab_class(size_t size) {
    for (int i = 0; i < SLAB_CLASSES; i++) {
        if (size <= slab_sizes[i]) return i;
    }
    return -1; /* too large for slab */
}

ce_mem_pool_t* ce_mem_pool_create(size_t total_size) {
    ce_mem_pool_t* pool = (ce_mem_pool_t*)calloc(1, sizeof(ce_mem_pool_t));
    if (!pool) return NULL;

    pool->base = aligned_alloc(CE_CACHE_LINE_SIZE,
                               (total_size + CE_CACHE_LINE_SIZE - 1) & ~(CE_CACHE_LINE_SIZE - 1));
    if (!pool->base) {
        free(pool);
        return NULL;
    }

    pool->total_size = total_size;
    pool->used = 0;

    /* Initialize slab free lists as empty */
    for (int i = 0; i < SLAB_CLASSES; i++) {
        pool->classes[i].free_list = NULL;
        pool->classes[i].block_size = slab_sizes[i];
        pool->classes[i].alloc_count = 0;
        pool->classes[i].free_count = 0;
    }

    return pool;
}

void ce_mem_pool_destroy(ce_mem_pool_t* pool) {
    if (!pool) return;
    free(pool->base);
    free(pool);
}

void* ce_mem_pool_alloc(ce_mem_pool_t* pool, size_t size) {
    if (!pool) return NULL;

    int cls = find_slab_class(size);
    if (cls >= 0) {
        slab_class_t* sc = &pool->classes[cls];
        if (sc->free_list) {
            /* Reuse from free list */
            void* ptr = sc->free_list;
            sc->free_list = sc->free_list->next;
            sc->alloc_count++;
            return ptr;
        }
    }

    /* Bump allocator for slab or large allocations */
    size_t aligned_size = (size + CE_CACHE_LINE_SIZE - 1) & ~(CE_CACHE_LINE_SIZE - 1);
    if (pool->used + aligned_size > pool->total_size) {
        return NULL; /* pool exhausted */
    }

    void* ptr = (char*)pool->base + pool->used;
    pool->used += aligned_size;

    if (cls >= 0) {
        pool->classes[cls].alloc_count++;
    }

    return ptr;
}

void ce_mem_pool_free(ce_mem_pool_t* pool, void* ptr) {
    if (!pool || !ptr) return;

    /* Check if pointer is within pool */
    if ((char*)ptr < (char*)pool->base || (char*)ptr >= (char*)pool->base + pool->total_size) return;

    /* Try to find which slab class this belongs to by checking alignment */
    for (int i = 0; i < SLAB_CLASSES; i++) {
        size_t blk = slab_sizes[i];
        uintptr_t offset = (uintptr_t)ptr - (uintptr_t)pool->base;
        if (offset % blk == 0) {
            slab_block_t* block = (slab_block_t*)ptr;
            block->next = pool->classes[i].free_list;
            pool->classes[i].free_list = block;
            pool->classes[i].free_count++;
            return;
        }
    }
}

void ce_mem_pool_reset(ce_mem_pool_t* pool) {
    if (!pool) return;
    pool->used = 0;
    for (int i = 0; i < SLAB_CLASSES; i++) {
        pool->classes[i].free_list = NULL;
        pool->classes[i].alloc_count = 0;
        pool->classes[i].free_count = 0;
    }
}

/* ---- Aligned allocation (global, tracked) ---- */

void* ce_aligned_alloc(size_t alignment, size_t size) {
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    /* Round size up to alignment */
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    void* ptr = NULL;
#if defined(_POSIX_VERSION) && _POSIX_VERSION >= 200112L
    if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
        return NULL;
    }
#else
    ptr = aligned_alloc(alignment, aligned_size);
#endif
    if (!ptr) return NULL;

    ce_tracker_record_alloc(ptr, aligned_size, NULL);
    return ptr;
}

void ce_aligned_free(void* ptr) {
    if (!ptr) return;
    ce_tracker_record_free(ptr);
    free(ptr);
}
