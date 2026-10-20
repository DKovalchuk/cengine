#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ---- Tracked allocation record ---- */
typedef struct alloc_record {
    void*                ptr;
    size_t               size;
    struct alloc_record* next;
} alloc_record_t;

/* Simple hash table for allocation tracking */
#define TRACKER_BUCKETS 1024

typedef struct {
    alloc_record_t* buckets[TRACKER_BUCKETS];
    int64_t current_bytes;
    int64_t peak_bytes;
    int64_t total_allocs;
    int64_t total_frees;
    int64_t alloc_bytes_interval;
    int64_t free_bytes_interval;
    int64_t tensor_count;
    int64_t tensor_total_bytes;

    /* For growth rate calculation */
    int64_t prev_current_bytes;
    struct timespec prev_time;
    double growth_rate;

    bool initialized;
} tracker_state_t;

static tracker_state_t g_tracker = {0};

static uint32_t hash_ptr(void* ptr) {
    uintptr_t v = (uintptr_t)ptr;
    v = (v >> 3) ^ v;
    return (uint32_t)(v % TRACKER_BUCKETS);
}

ce_status_t ce_tracker_init(void) {
    if (g_tracker.initialized) return CE_OK;
    memset(&g_tracker, 0, sizeof(g_tracker));
    clock_gettime(CLOCK_MONOTONIC, &g_tracker.prev_time);
    g_tracker.initialized = true;
    return CE_OK;
}

void ce_tracker_shutdown(void) {
    if (!g_tracker.initialized) return;
    g_tracker.initialized = false;
}

void ce_tracker_record_alloc(void* ptr, size_t size, const char* tag) {
    (void)tag;
    if (!ptr || !g_tracker.initialized) return;

    alloc_record_t* rec = (alloc_record_t*)malloc(sizeof(alloc_record_t));
    if (!rec) return;

    rec->ptr = ptr;
    rec->size = size;

    uint32_t bucket = hash_ptr(ptr);
    rec->next = g_tracker.buckets[bucket];
    g_tracker.buckets[bucket] = rec;

    g_tracker.current_bytes += (int64_t)size;
    g_tracker.total_allocs++;
    g_tracker.alloc_bytes_interval += (int64_t)size;

    if (g_tracker.current_bytes > g_tracker.peak_bytes) {
        g_tracker.peak_bytes = g_tracker.current_bytes;
    }
}

void ce_tracker_record_free(void* ptr) {
    if (!ptr || !g_tracker.initialized) return;

    uint32_t bucket = hash_ptr(ptr);
    alloc_record_t** pp = &g_tracker.buckets[bucket];

    while (*pp) {
        if ((*pp)->ptr == ptr) {
            alloc_record_t* rec = *pp;
            *pp = rec->next;

            g_tracker.current_bytes -= (int64_t)rec->size;
            g_tracker.total_frees++;
            g_tracker.free_bytes_interval += (int64_t)rec->size;

            free(rec);
            return;
        }
        pp = &(*pp)->next;
    }
    /* ptr not found — double free or untracked */
}

void ce_tracker_update_tensor(int64_t count_delta, int64_t bytes_delta) {
    if (!g_tracker.initialized) return;
    g_tracker.tensor_count += count_delta;
    g_tracker.tensor_total_bytes += bytes_delta;
}

ce_memory_stats_t ce_tracker_get_stats(void) {
    ce_memory_stats_t stats = {0};
    if (!g_tracker.initialized) return stats;

    /* Calculate growth rate */
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double dt = (now.tv_sec - g_tracker.prev_time.tv_sec) +
                (now.tv_nsec - g_tracker.prev_time.tv_nsec) / 1e9;

    if (dt > 0.1) { /* update rate at most every 100ms */
        g_tracker.growth_rate = (double)(g_tracker.current_bytes - g_tracker.prev_current_bytes) / dt;
        g_tracker.prev_current_bytes = g_tracker.current_bytes;
        g_tracker.prev_time = now;
    }

    stats.current_bytes   = g_tracker.current_bytes;
    stats.peak_bytes      = g_tracker.peak_bytes;
    stats.total_allocs    = g_tracker.total_allocs;
    stats.total_frees     = g_tracker.total_frees;
    stats.active_allocs   = g_tracker.total_allocs - g_tracker.total_frees;
    stats.alloc_bytes_rate = g_tracker.alloc_bytes_interval;
    stats.free_bytes_rate  = g_tracker.free_bytes_interval;
    stats.growth_rate      = g_tracker.growth_rate;
    stats.tensor_count     = g_tracker.tensor_count;
    stats.tensor_total_bytes = g_tracker.tensor_total_bytes;

    /* Fragmentation heuristic */
    if (g_tracker.peak_bytes > 0) {
        stats.fragmentation = 1.0 - (double)g_tracker.current_bytes / g_tracker.peak_bytes;
    }

    return stats;
}

void ce_tracker_reset_rates(void) {
    if (!g_tracker.initialized) return;
    g_tracker.alloc_bytes_interval = 0;
    g_tracker.free_bytes_interval = 0;
}

int ce_tracker_check_leaks(void) {
    if (!g_tracker.initialized) return 0;
    return (int)(g_tracker.total_allocs - g_tracker.total_frees);
}
