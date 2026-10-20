#include "cengine.h"
#include "threadpool.h"

#include <string.h>

/* Lock-free SPSC ring buffer.
 * Single producer pushes to head, single consumer pops from tail.
 * Uses relaxed atomics — safe for SPSC without fences on x86. */

int ce_ring_push(ce_ring_buffer_t* ring, const ce_metric_t* metric) {
    if (!ring || !metric) return -1;

    uint64_t head = atomic_load_explicit(&ring->head, memory_order_relaxed);
    uint64_t tail = atomic_load_explicit(&ring->tail, memory_order_acquire);

    if (head - tail >= CE_RING_CAPACITY) {
        return -1;  /* full */
    }

    ring->buffer[head & CE_RING_MASK] = *metric;
    atomic_store_explicit(&ring->head, head + 1, memory_order_release);
    return 0;
}

int ce_ring_pop(ce_ring_buffer_t* ring, ce_metric_t* out) {
    if (!ring || !out) return -1;

    uint64_t tail = atomic_load_explicit(&ring->tail, memory_order_relaxed);
    uint64_t head = atomic_load_explicit(&ring->head, memory_order_acquire);

    if (tail == head) {
        return -1;  /* empty */
    }

    *out = ring->buffer[tail & CE_RING_MASK];
    atomic_store_explicit(&ring->tail, tail + 1, memory_order_release);
    return 0;
}

int ce_ring_batch_pop(ce_ring_buffer_t* ring, ce_metric_t* out, int max_count) {
    if (!ring || !out || max_count <= 0) return 0;

    uint64_t tail = atomic_load_explicit(&ring->tail, memory_order_relaxed);
    uint64_t head = atomic_load_explicit(&ring->head, memory_order_acquire);

    uint64_t available = head - tail;
    if (available == 0) return 0;

    int count = (int)((available < (uint64_t)max_count)
                      ? available : (uint64_t)max_count);

    for (int i = 0; i < count; i++) {
        out[i] = ring->buffer[(tail + (uint64_t)i) & CE_RING_MASK];
    }

    atomic_store_explicit(&ring->tail, tail + (uint64_t)count,
                          memory_order_release);
    return count;
}
