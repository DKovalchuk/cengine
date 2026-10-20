#include "cengine.h"
#include "threadpool.h"

#include <string.h>
#include <immintrin.h>

ce_status_t ce_barrier_init(ce_barrier_t* barrier, int num_threads) {
    if (!barrier || num_threads <= 0) return CE_ERR_INVALID_ARG;
    atomic_store(&barrier->count, 0);
    atomic_store(&barrier->generation, 0);
    barrier->num_threads = num_threads;
    return CE_OK;
}

void ce_barrier_wait(ce_barrier_t* barrier) {
    int gen = atomic_load(&barrier->generation);

    int arrived = atomic_fetch_add(&barrier->count, 1) + 1;

    if (arrived == barrier->num_threads) {
        /* Last thread to arrive: reset count and advance generation */
        atomic_store(&barrier->count, 0);
        atomic_store(&barrier->generation, gen + 1);
    } else {
        /* Spin until generation changes */
        while (atomic_load(&barrier->generation) == gen) {
            _mm_pause();
        }
    }
}

void ce_barrier_destroy(ce_barrier_t* barrier) {
    if (!barrier) return;
    /* Nothing to free — barrier is typically stack/embedded allocated */
    (void)barrier;
}
