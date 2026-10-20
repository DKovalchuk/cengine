/*
 * dataset.c — ALU sample lifecycle + bit/carry helpers
 */
#include "data.h"
#include "tensor.h"
#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Bit helpers                                                         */
/* ------------------------------------------------------------------ */

void ce_bits_le(uint64_t value, int n_bits, float* out) {
    for (int i = 0; i < n_bits; i++) {
        out[i] = (float)((value >> i) & 1);
    }
}

void ce_compute_add_carry(uint64_t a, uint64_t b, int n_bits, float* out) {
    int c = 0;
    for (int i = 0; i < n_bits; i++) {
        int a_bit = (a >> i) & 1;
        int b_bit = (b >> i) & 1;
        int sum = a_bit + b_bit + c;
        c = (sum >= 2) ? 1 : 0;
        out[i] = (float)c;
    }
}

void ce_compute_sub_carry(uint64_t a, uint64_t b, int n_bits, float* out) {
    int c = 1;  /* +1 for two's complement */
    for (int i = 0; i < n_bits; i++) {
        int a_bit = (a >> i) & 1;
        int b_bit = 1 - ((b >> i) & 1);  /* invert b */
        int sum = a_bit + b_bit + c;
        c = (sum >= 2) ? 1 : 0;
        out[i] = (float)c;
    }
}

/* ------------------------------------------------------------------ */
/* Sample creation / destruction                                       */
/* ------------------------------------------------------------------ */

static ce_tensor_t* make_1d_tensor(int n, const char* name) {
    int64_t shape[1] = { n };
    ce_tensor_t* t = ce_tensor_zeros(shape, 1);
    if (t) {
        snprintf(t->name, sizeof(t->name), "%s", name);
    }
    return t;
}

/*
 * Create a sample from raw integers.  All bit vectors are LSB-first.
 */
static ce_alu_sample_t* create_sample(uint64_t a, uint64_t b, int op,
                                      int n_bits) {
    ce_alu_sample_t* s = calloc(1, sizeof(*s));
    if (!s) return NULL;

    s->n_bits = n_bits;
    s->op     = op;

    s->a_bits      = make_1d_tensor(n_bits, "a_bits");
    s->b_bits      = make_1d_tensor(n_bits, "b_bits");
    s->op_code     = make_1d_tensor(3, "op_code");
    s->result_bits = make_1d_tensor(n_bits, "result_bits");
    s->carry_bits  = make_1d_tensor(n_bits, "carry_bits");

    if (!s->a_bits || !s->b_bits || !s->op_code ||
        !s->result_bits || !s->carry_bits) {
        ce_alu_sample_destroy(s);
        return NULL;
    }

    /* Fill bit vectors */
    ce_bits_le(a, n_bits, s->a_bits->data);
    ce_bits_le(b, n_bits, s->b_bits->data);

    /* op_code as 3-element vector (binary encoding) */
    s->op_code->data[0] = (float)(op & 1);
    s->op_code->data[1] = (float)((op >> 1) & 1);
    s->op_code->data[2] = (float)((op >> 2) & 1);

    /* Result */
    uint64_t max_val = (n_bits >= 64) ? ~0ULL : (1ULL << n_bits);
    uint64_t result;
    switch (op) {
        case CE_OP_ADD: result = (a + b) % max_val; break;
        case CE_OP_SUB: result = (a - b) % max_val; break;
        case CE_OP_AND: result = a & b;             break;
        case CE_OP_XOR: result = a ^ b;             break;
        case CE_OP_OR:  result = a | b;             break;
        default:        result = 0;                  break;
    }
    ce_bits_le(result, n_bits, s->result_bits->data);

    /* Carry bits */
    switch (op) {
        case CE_OP_ADD:
            ce_compute_add_carry(a, b, n_bits, s->carry_bits->data);
            break;
        case CE_OP_SUB:
            ce_compute_sub_carry(a, b, n_bits, s->carry_bits->data);
            break;
        default:
            /* AND/XOR: carry always 0 (tensor already zeroed) */
            break;
    }

    return s;
}

void ce_alu_sample_destroy(ce_alu_sample_t* sample) {
    if (!sample) return;
    if (sample->a_bits)      ce_tensor_release(sample->a_bits);
    if (sample->b_bits)      ce_tensor_release(sample->b_bits);
    if (sample->op_code)     ce_tensor_release(sample->op_code);
    if (sample->result_bits) ce_tensor_release(sample->result_bits);
    if (sample->carry_bits)  ce_tensor_release(sample->carry_bits);
    free(sample);
}

/* ------------------------------------------------------------------ */
/* Batch                                                               */
/* ------------------------------------------------------------------ */

ce_batch_t* ce_batch_create(int capacity) {
    ce_batch_t* b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->samples  = calloc((size_t)capacity, sizeof(ce_alu_sample_t*));
    if (!b->samples) { free(b); return NULL; }
    b->count    = 0;
    b->capacity = capacity;
    return b;
}

static int ce_batch_append(ce_batch_t* b, ce_alu_sample_t* s) {
    if (!b || !s) return -1;
    if (b->count >= b->capacity) return -1;
    b->samples[b->count++] = s;
    return 0;
}

void ce_batch_destroy(ce_batch_t* batch) {
    if (!batch) return;
    for (int i = 0; i < batch->count; i++) {
        ce_alu_sample_destroy(batch->samples[i]);
    }
    free(batch->samples);
    free(batch);
}

/*
 * Public: generate a single sample using the generator's RNG.
 * Used internally by alu_generator.c via the create_sample helper above.
 */
ce_alu_sample_t* ce_alu_gen_sample(ce_alu_gen_t* gen) {
    if (!gen) return NULL;

    int n_bits = (int)ce_rand_range(&gen->rng_state,
                                    (uint64_t)(gen->max_bits - gen->min_bits + 1))
                 + gen->min_bits;
    int op_idx = (int)ce_rand_range(&gen->rng_state, (uint64_t)gen->n_ops);
    int op     = gen->ops[op_idx];

    uint64_t max_val = (n_bits >= 64) ? ~0ULL : (1ULL << n_bits);
    uint64_t a = ce_xorshift64star(&gen->rng_state) % max_val;
    uint64_t b = ce_xorshift64star(&gen->rng_state) % max_val;

    return create_sample(a, b, op, n_bits);
}

ce_batch_t* ce_alu_gen_batch(ce_alu_gen_t* gen, int batch_size) {
    if (!gen || batch_size <= 0) return NULL;

    ce_batch_t* batch = ce_batch_create(batch_size);
    if (!batch) return NULL;

    for (int i = 0; i < batch_size; i++) {
        ce_alu_sample_t* s = ce_alu_gen_sample(gen);
        if (!s) { ce_batch_destroy(batch); return NULL; }
        ce_batch_append(batch, s);
    }
    return batch;
}
