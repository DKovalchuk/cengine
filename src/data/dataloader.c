/*
 * dataloader.c — Async dataloader with double-buffered prefetch
 *
 * A background pthread generates batches into the inactive buffer while
 * the consumer reads from the active one.  Atomic flags coordinate handoff.
 */
#include "data.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Prefetch thread                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    ce_dataloader_t* dl;
    int              buffer_idx;  /* which prefetch_buffer to fill */
} prefetch_arg_t;

static void* prefetch_worker(void* arg) {
    prefetch_arg_t* pa = (prefetch_arg_t*)arg;
    ce_dataloader_t* dl = pa->dl;
    int idx = pa->buffer_idx;
    free(pa);

    /* Destroy old batch if present */
    if (dl->prefetch_buffer[idx]) {
        ce_batch_destroy(dl->prefetch_buffer[idx]);
        dl->prefetch_buffer[idx] = NULL;
    }

    /* Generate a fresh batch from the generator */
    ce_batch_t* batch = ce_alu_gen_batch(dl->generator, dl->batch_size);

    /* Publish */
    dl->prefetch_buffer[idx] = batch;
    atomic_store(&dl->ready_buffer, idx);

    return NULL;
}

/* Start the background prefetch for the given buffer index */
static void start_prefetch(ce_dataloader_t* dl, int idx) {
    prefetch_arg_t* pa = malloc(sizeof(*pa));
    if (!pa) {
        /* Fallback: mark buffer ready (will be NULL → caller handles) */
        atomic_store(&dl->ready_buffer, idx);
        return;
    }
    pa->dl = dl;
    pa->buffer_idx = idx;

    dl->thread_active = true;
    int rc = pthread_create(&dl->prefetch_thread, NULL, prefetch_worker, pa);
    if (rc != 0) {
        free(pa);
        /* Fallback: generate synchronously */
        if (dl->prefetch_buffer[idx]) {
            ce_batch_destroy(dl->prefetch_buffer[idx]);
        }
        dl->prefetch_buffer[idx] = ce_alu_gen_batch(dl->generator,
                                                     dl->batch_size);
        atomic_store(&dl->ready_buffer, idx);
        dl->thread_active = false;
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

ce_dataloader_t* ce_dataloader_create(ce_alu_gen_t* gen, int batch_size,
                                      int samples_per_epoch, int64_t seed) {
    if (!gen || batch_size <= 0 || samples_per_epoch <= 0) return NULL;

    ce_dataloader_t* dl = calloc(1, sizeof(*dl));
    if (!dl) return NULL;

    dl->generator        = gen;
    dl->batch_size       = batch_size;
    dl->samples_per_epoch = samples_per_epoch;
    dl->base_seed        = seed;
    dl->epoch            = 0;
    dl->prefetch_buffer[0] = NULL;
    dl->prefetch_buffer[1] = NULL;
    atomic_store(&dl->ready_buffer, -1);
    atomic_store(&dl->stop_flag, false);
    dl->thread_active    = false;

    /* Kick off first prefetch immediately */
    start_prefetch(dl, 0);

    return dl;
}

ce_batch_t* ce_dataloader_next(ce_dataloader_t* dl) {
    if (!dl) return NULL;

    /* Wait for the ready buffer */
    int ready;
    while ((ready = atomic_load(&dl->ready_buffer)) < 0) {
        if (atomic_load(&dl->stop_flag)) return NULL;
        /* Spin — in practice the batch is ready almost instantly */
        struct timespec ts = { .tv_nsec = 1000 }; /* 1 µs */
        nanosleep(&ts, NULL);
    }

    /* If a background thread finished, join it */
    if (dl->thread_active) {
        pthread_join(dl->prefetch_thread, NULL);
        dl->thread_active = false;
    }

    /* Take the ready batch */
    ce_batch_t* batch = dl->prefetch_buffer[ready];
    dl->prefetch_buffer[ready] = NULL;
    atomic_store(&dl->ready_buffer, -1);

    /* Track progress */
    dl->epoch++;

    /* Start prefetching the NEXT batch into the other buffer */
    int next_buf = 1 - ready;
    start_prefetch(dl, next_buf);

    return batch;
}

void ce_dataloader_reset(ce_dataloader_t* dl) {
    if (!dl) return;

    /* Stop any running thread */
    atomic_store(&dl->stop_flag, true);
    if (dl->thread_active) {
        pthread_join(dl->prefetch_thread, NULL);
        dl->thread_active = false;
    }
    atomic_store(&dl->stop_flag, false);

    /* Free existing buffers */
    for (int i = 0; i < 2; i++) {
        if (dl->prefetch_buffer[i]) {
            ce_batch_destroy(dl->prefetch_buffer[i]);
            dl->prefetch_buffer[i] = NULL;
        }
    }
    atomic_store(&dl->ready_buffer, -1);

    /* Re-seed the generator for new epoch */
    dl->epoch = 0;
    dl->generator->rng_state = (uint64_t)dl->base_seed;
    if (dl->generator->rng_state == 0) {
        dl->generator->rng_state = 0x123456789ABCDEF0ULL;
    }
    for (int i = 0; i < 8; i++) {
        ce_xorshift64(&dl->generator->rng_state);
    }

    /* Restart prefetch */
    start_prefetch(dl, 0);
}

void ce_dataloader_destroy(ce_dataloader_t* dl) {
    if (!dl) return;

    /* Signal stop and join thread */
    atomic_store(&dl->stop_flag, true);
    if (dl->thread_active) {
        pthread_join(dl->prefetch_thread, NULL);
        dl->thread_active = false;
    }

    /* Free buffers */
    for (int i = 0; i < 2; i++) {
        if (dl->prefetch_buffer[i]) {
            ce_batch_destroy(dl->prefetch_buffer[i]);
        }
    }

    free(dl);
}
