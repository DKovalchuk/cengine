/*
 * alu_generator.c — ALU generator creation and seed management
 *
 * Core generation logic (ce_alu_gen_sample, ce_alu_gen_batch) lives in
 * dataset.c alongside the sample lifecycle.  This file handles generator
 * construction and configuration.
 */
#include "data.h"
#include <stdlib.h>
#include <string.h>

ce_alu_gen_t* ce_alu_gen_create(int min_bits, int max_bits,
                                const int* ops, int n_ops, int64_t seed) {
    if (min_bits < 1 || max_bits < min_bits || !ops || n_ops < 1 ||
        n_ops > CE_NUM_OPS) {
        return NULL;
    }

    ce_alu_gen_t* gen = calloc(1, sizeof(*gen));
    if (!gen) return NULL;

    gen->min_bits = min_bits;
    gen->max_bits = max_bits;
    gen->n_ops    = n_ops;
    gen->seed     = seed;

    memcpy(gen->ops, ops, (size_t)n_ops * sizeof(int));

    /* Seed the PRNG — avoid zero state */
    gen->rng_state = (uint64_t)seed;
    if (gen->rng_state == 0) {
        gen->rng_state = 0x123456789ABCDEF0ULL;
    }
    /* Warm up the generator */
    for (int i = 0; i < 8; i++) {
        ce_xorshift64(&gen->rng_state);
    }

    return gen;
}

void ce_alu_gen_destroy(ce_alu_gen_t* gen) {
    /* The struct owns nothing else (ops[] is inline) — plain free.
     * Callers must have stopped every reader first (e.g. destroyed the
     * dataloader, which borrows this pointer). */
    free(gen);
}
