#ifndef CENGINE_THREADPOOL_H
#define CENGINE_THREADPOOL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include "cengine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Thread Pool
 * ======================================================================== */

typedef void (*ce_task_func_t)(void* arg);

typedef struct {
    ce_task_func_t func;
    void*          arg;
} ce_task_t;

typedef struct ce_threadpool ce_threadpool_t;

ce_threadpool_t* ce_threadpool_create(int num_workers);
ce_status_t      ce_threadpool_submit(ce_threadpool_t* pool,
                                      ce_task_func_t func, void* arg);
ce_status_t      ce_threadpool_wait_all(ce_threadpool_t* pool);
void             ce_threadpool_destroy(ce_threadpool_t* pool);
int              ce_threadpool_get_num_workers(ce_threadpool_t* pool);

/* ========================================================================
 * Spin Barrier
 * ======================================================================== */

typedef struct {
    _Alignas(64) _Atomic(int) count;
    _Alignas(64) _Atomic(int) generation;
    int num_threads;
} ce_barrier_t;

ce_status_t ce_barrier_init(ce_barrier_t* barrier, int num_threads);
void        ce_barrier_wait(ce_barrier_t* barrier);
void        ce_barrier_destroy(ce_barrier_t* barrier);

/* ========================================================================
 * Lock-Free SPSC Ring Buffer
 * ======================================================================== */

#define CE_RING_CAPACITY 4096  /* must be power of 2 */
#define CE_RING_MASK     (CE_RING_CAPACITY - 1)

typedef struct {
    uint8_t  metric_type;
    uint8_t  flags;
    uint16_t _pad;
    uint32_t epoch;
    uint64_t timestamp_ns;
    union {
        float    f32;
        float    f32_arr[8];
        int64_t  i64;
    } value;
} ce_metric_t;

typedef struct {
    _Alignas(64) _Atomic(uint64_t) head;
    _Alignas(64) _Atomic(uint64_t) tail;
    _Alignas(64) ce_metric_t buffer[CE_RING_CAPACITY];
} ce_ring_buffer_t;

/* SPSC: single producer calls push, single consumer calls pop/batch_pop */
int ce_ring_push(ce_ring_buffer_t* ring, const ce_metric_t* metric);
int ce_ring_pop(ce_ring_buffer_t* ring, ce_metric_t* out);
int ce_ring_batch_pop(ce_ring_buffer_t* ring, ce_metric_t* out, int max_count);

/* ========================================================================
 * Load Balancer
 * ======================================================================== */

typedef struct {
    int   min_workers;
    int   max_workers;
    float target_cpu_util;
    int   current_workers;

    /* previous CPU jiffies for delta calculation */
    uint64_t prev_idle;
    uint64_t prev_total;
} ce_load_balancer_t;

ce_load_balancer_t* ce_load_balancer_create(int min_workers, int max_workers,
                                            float target_cpu_util);
int  ce_load_balancer_adjust(ce_load_balancer_t* lb);
void ce_load_balancer_destroy(ce_load_balancer_t* lb);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_THREADPOOL_H */
