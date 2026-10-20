#ifndef CENGINE_DATA_H
#define CENGINE_DATA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>
#include "cengine.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- ALU operation codes (match data.py) ---- */
#define CE_OP_ADD  0
#define CE_OP_SUB  1
#define CE_OP_AND  2
#define CE_OP_XOR  3
#define CE_OP_OR   4
#define CE_NUM_OPS 5

/* ---- ALU Sample ---- */
typedef struct {
    ce_tensor_t* a_bits;      /* [n_bits] — LSB-first */
    ce_tensor_t* b_bits;      /* [n_bits] */
    ce_tensor_t* op_code;     /* [3] — 3-bit binary encoding */
    ce_tensor_t* result_bits; /* [n_bits] */
    ce_tensor_t* carry_bits;  /* [n_bits] */
    int          n_bits;
    int          op;          /* operation code (0-4) */
} ce_alu_sample_t;

/* ---- Batch ---- */
typedef struct {
    ce_alu_sample_t** samples;
    int count;
    int capacity;
} ce_batch_t;

/* ---- ALU Generator ---- */
typedef struct {
    int      min_bits;
    int      max_bits;
    int      ops[CE_NUM_OPS];
    int      n_ops;
    int64_t  seed;
    uint64_t rng_state;  /* xorshift64* state */
} ce_alu_gen_t;

ce_alu_gen_t*    ce_alu_gen_create(int min_bits, int max_bits,
                                   const int* ops, int n_ops, int64_t seed);
ce_alu_sample_t* ce_alu_gen_sample(ce_alu_gen_t* gen);
ce_batch_t*      ce_alu_gen_batch(ce_alu_gen_t* gen, int batch_size);
void             ce_alu_sample_destroy(ce_alu_sample_t* sample);
void             ce_batch_destroy(ce_batch_t* batch);

/* ---- Curriculum Stage ---- */
typedef struct {
    const char* name;
    int         min_bits;
    int         max_bits;
    int         ops[CE_NUM_OPS];
    int         n_ops;
} ce_curriculum_stage_t;

/* ---- Curriculum Manager ---- */
typedef struct {
    ce_curriculum_stage_t* stages;
    int   n_stages;
    int   current_stage;
    int   epochs_in_stage;
    int   consecutive_good;
    float loss_threshold;
    int   patience;
    int   max_epochs_per_stage;
} ce_curriculum_t;

ce_curriculum_t*  ce_curriculum_create(float loss_threshold, int patience,
                                       int max_epochs, int max_bits);
int               ce_curriculum_update(ce_curriculum_t* c, float epoch_loss);
ce_alu_gen_t*     ce_curriculum_make_generator(ce_curriculum_t* c, int64_t seed);
const char*       ce_curriculum_stage_name(const ce_curriculum_t* c);
bool              ce_curriculum_is_final(const ce_curriculum_t* c);
void              ce_curriculum_destroy(ce_curriculum_t* c);

/* ---- DataLoader (async prefetch) ---- */
typedef struct {
    ce_alu_gen_t*  generator;
    int            batch_size;
    int            samples_per_epoch;
    int64_t        base_seed;
    int64_t        epoch;

    /* Double-buffer prefetch */
    ce_batch_t*    prefetch_buffer[2];
    atomic_int     ready_buffer;     /* index of ready buffer, -1 = none */
    pthread_t      prefetch_thread;
    atomic_bool    stop_flag;
    bool           thread_active;
} ce_dataloader_t;

ce_dataloader_t* ce_dataloader_create(ce_alu_gen_t* gen, int batch_size,
                                      int samples_per_epoch, int64_t seed);
ce_batch_t*      ce_dataloader_next(ce_dataloader_t* dl);
void             ce_dataloader_reset(ce_dataloader_t* dl);
void             ce_dataloader_destroy(ce_dataloader_t* dl);

/* ---- Bit helpers (exposed for testing) ---- */
void  ce_bits_le(uint64_t value, int n_bits, float* out);
void  ce_compute_add_carry(uint64_t a, uint64_t b, int n_bits, float* out);
void  ce_compute_sub_carry(uint64_t a, uint64_t b, int n_bits, float* out);

/* ---- PRNG ---- */
static inline uint64_t ce_xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* xorshift64* — better quality */
static inline uint64_t ce_xorshift64star(uint64_t* state) {
    return ce_xorshift64(state) * 0x2545F4914F6CDD1DULL;
}

/* Uniform random int in [0, range) */
static inline uint64_t ce_rand_range(uint64_t* state, uint64_t range) {
    if (range == 0) return 0;
    /* Lemire's nearly-divisionless method */
    uint64_t x = ce_xorshift64star(state);
    __uint128_t m = (__uint128_t)x * (__uint128_t)range;
    uint64_t l = (uint64_t)m;
    if (l < range) {
        uint64_t t = (-range) % range;
        while (l < t) {
            x = ce_xorshift64star(state);
            m = (__uint128_t)x * (__uint128_t)range;
            l = (uint64_t)m;
        }
    }
    return (uint64_t)(m >> 64);
}

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_DATA_H */
