/* linear.c — Fully connected layer: Y = X @ W^T + b */
#include "layers.h"
#include "autograd.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* State saved for backward pass */
typedef struct {
    ce_tensor_t* input_cache;
    ce_tensor_t* weight;
    ce_tensor_t* bias;
    ce_tensor_t* grad_w;
    ce_tensor_t* grad_b;
} linear_backward_state_t;

/* Release the references taken at record() before freeing the state —
 * the tape owns the state and cannot know its layout. */
static void linear_state_free(void* state) {
    linear_backward_state_t* s = (linear_backward_state_t*)state;
    if (s->input_cache) ce_tensor_release(s->input_cache);
    if (s->weight)      ce_tensor_release(s->weight);
    if (s->bias)        ce_tensor_release(s->bias);
    if (s->grad_w)      ce_tensor_release(s->grad_w);
    if (s->grad_b)      ce_tensor_release(s->grad_b);
    free(s);
}

/* Backward function registered with the tape */
static void linear_tape_backward(void* state, ce_tensor_t* grad_output) {
    linear_backward_state_t* s = (linear_backward_state_t*)state;
    if (!s || !grad_output) return;

    ce_status_t st;

    /*
     * grad_w += grad_output^T @ input_cache
     *   [out, in] += [out, batch] @ [batch, in]
     *   = matmul(grad_output, input_cache, tmp, CE_TRANSPOSE_A)
     */
    {
        int64_t wshape[2] = { s->weight->shape[0], s->weight->shape[1] };
        ce_tensor_t* tmp = ce_tensor_zeros(wshape, 2);
        if (tmp) {
            st = ce_matmul(grad_output, s->input_cache, tmp, CE_TRANSPOSE_A);
            if (st == CE_OK) ce_axpy(1.0f, tmp, s->grad_w);
            ce_tensor_release(tmp);
        }
    }

    /*
     * grad_b += sum(grad_output, dim=0)
     *   Column-wise sum: [batch, out] → [out]
     */
    {
        ce_tensor_t* tmp_b = ce_tensor_zeros(&s->bias->shape[0], 1);
        if (tmp_b) {
            st = ce_get_default_backend()->sum(grad_output, tmp_b, 0);
            if (st == CE_OK) ce_axpy(1.0f, tmp_b, s->grad_b);
            ce_tensor_release(tmp_b);
        }
    }
}

/* ---- Xavier uniform initialization ---- */

static void xavier_uniform(ce_tensor_t* t, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    int64_t n = t->numel;
    /* Simple LCG-based deterministic init */
    uint32_t seed = 42;
    for (int64_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        float u = (float)(seed & 0x00FFFFFF) / (float)0x01000000;  /* [0, 1) */
        t->data[i] = (u * 2.0f - 1.0f) * limit;
    }
}

/* ---- Layer creation ---- */

ce_module_t* ce_linear_create(int in_features, int out_features,
                              const char* name) {
    if (in_features <= 0 || out_features <= 0) return NULL;

    ce_module_t* mod = ce_module_create(name ? name : "linear");
    if (!mod) return NULL;

    ce_linear_t* lin = (ce_linear_t*)calloc(1, sizeof(ce_linear_t));
    if (!lin) { ce_module_destroy(mod); return NULL; }

    lin->in_features  = in_features;
    lin->out_features = out_features;
    if (name) strncpy(lin->name, name, sizeof(lin->name) - 1);

    /* Allocate weight [out_features, in_features] */
    int64_t wshape[2] = { out_features, in_features };
    lin->weight = ce_tensor_create(wshape, 2, CE_FLOAT32);
    lin->grad_w = ce_tensor_zeros(wshape, 2);
    if (!lin->weight || !lin->grad_w) goto fail;

    xavier_uniform(lin->weight, in_features, out_features);

    /* Allocate bias [out_features] */
    int64_t bshape[1] = { out_features };
    lin->bias   = ce_tensor_create(bshape, 1, CE_FLOAT32);
    lin->grad_b = ce_tensor_zeros(bshape, 1);
    if (!lin->bias || !lin->grad_b) goto fail;

    mod->internal_state = lin;
    mod->forward  = ce_linear_forward;
    mod->backward = ce_linear_backward;

    /* Register parameters */
    ce_module_add_parameter(mod, lin->weight, lin->grad_w);
    ce_module_add_parameter(mod, lin->bias,   lin->grad_b);

    return mod;

fail:
    if (lin->weight) ce_tensor_release(lin->weight);
    if (lin->grad_w) ce_tensor_release(lin->grad_w);
    if (lin->bias)   ce_tensor_release(lin->bias);
    if (lin->grad_b) ce_tensor_release(lin->grad_b);
    free(lin);
    ce_module_destroy(mod);
    return NULL;
}

/* ---- Forward ---- */

ce_status_t ce_linear_forward(ce_module_t* mod,
                               ce_tensor_t* input, ce_tensor_t* output) {
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;

    ce_linear_t* lin = (ce_linear_t*)mod->internal_state;
    if (!lin) return CE_ERR_NOT_INITIALIZED;

    ce_status_t st;

    /* Cache input for backward */
    if (lin->input_cache) ce_tensor_release(lin->input_cache);
    lin->input_cache = ce_tensor_clone(input);
    if (!lin->input_cache) return CE_ERR_OUT_OF_MEMORY;

    /* output = input @ weight^T   [batch, in] @ [in, out] → [batch, out] */
    st = ce_matmul(input, lin->weight, output, CE_TRANSPOSE_B);
    if (st != CE_OK) return st;

    /* output += bias (broadcast across batch dimension) */
    int64_t batch = input->shape[0];
    int64_t out_f = lin->out_features;
    for (int64_t b = 0; b < batch; b++) {
        float* row = output->data + b * out_f;
        for (int64_t j = 0; j < out_f; j++) {
            row[j] += lin->bias->data[j];
        }
    }

    /* Record tape entry */
    ce_tape_t* tape = ce_get_tape();
    if (tape && tape->recording) {
        linear_backward_state_t* bs =
            (linear_backward_state_t*)calloc(1, sizeof(linear_backward_state_t));
        if (bs) {
            ce_tensor_retain(lin->input_cache);
            ce_tensor_retain(lin->weight);
            ce_tensor_retain(lin->bias);
            ce_tensor_retain(lin->grad_w);
            ce_tensor_retain(lin->grad_b);
            bs->input_cache = lin->input_cache;
            bs->weight      = lin->weight;
            bs->bias        = lin->bias;
            bs->grad_w      = lin->grad_w;
            bs->grad_b      = lin->grad_b;
            ce_tape_record(tape, output, linear_tape_backward, bs, 1,
                           linear_state_free);
        }
    }

    return CE_OK;
}

/* ---- Backward (standalone, non-tape) ---- */

ce_status_t ce_linear_backward(ce_module_t* mod,
                                ce_tensor_t* grad_output,
                                ce_tensor_t* grad_input) {
    if (!mod || !grad_output) return CE_ERR_INVALID_ARG;

    ce_linear_t* lin = (ce_linear_t*)mod->internal_state;
    if (!lin || !lin->input_cache) return CE_ERR_NOT_INITIALIZED;

    ce_status_t st;

    /* grad_w += grad_output^T @ input_cache */
    {
        int64_t wshape[2] = { lin->weight->shape[0], lin->weight->shape[1] };
        ce_tensor_t* tmp = ce_tensor_zeros(wshape, 2);
        if (!tmp) return CE_ERR_OUT_OF_MEMORY;
        st = ce_matmul(grad_output, lin->input_cache, tmp, CE_TRANSPOSE_A);
        if (st == CE_OK) ce_axpy(1.0f, tmp, lin->grad_w);
        ce_tensor_release(tmp);
    }

    /* grad_b += sum(grad_output, dim=0) */
    {
        ce_tensor_t* tmp_b = ce_tensor_zeros(&lin->bias->shape[0], 1);
        if (!tmp_b) return CE_ERR_OUT_OF_MEMORY;
        st = ce_get_default_backend()->sum(grad_output, tmp_b, 0);
        if (st == CE_OK) ce_axpy(1.0f, tmp_b, lin->grad_b);
        ce_tensor_release(tmp_b);
    }

    /* grad_input = grad_output @ weight */
    if (grad_input) {
        st = ce_matmul(grad_output, lin->weight, grad_input, CE_TRANSPOSE_NONE);
        if (st != CE_OK) return st;
    }

    return CE_OK;
}
