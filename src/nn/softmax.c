/* softmax.c — Softmax and Gumbel-Softmax (STE) activation layers */
#include "layers.h"
#include "autograd.h"
#include <stdlib.h>

typedef struct {
    ce_tensor_t* output_ref;
    float temperature;       /* 0.0 for standard softmax */
} softmax_state_t;

/* Layer state for the production (non-tape) path — mirrors sigmoid:
 * forward caches its output, backward needs it for the JVP. */
typedef struct {
    ce_activation_t act_type;
    float temperature;
    ce_tensor_t* output_cache;
} softmax_layer_state_t;

/* Backward: Jacobian-vector product for softmax
 *   grad_input_i = y_i * (grad_i - sum_j(grad_j * y_j))
 * Also used for Gumbel-Softmax STE (same formula). */
static void softmax_tape_backward(void* state, ce_tensor_t* grad_output) {
    softmax_state_t* s = (softmax_state_t*)state;
    if (!s || !grad_output || !s->output_ref) return;

    /*
     * Compute in-place:
     *   dot = sum(grad_output * output) per row
     *   grad_input = output * (grad_output - dot)
     */
    int64_t batch   = (grad_output->ndim >= 2) ? grad_output->shape[0] : 1;
    int64_t classes = grad_output->numel / batch;
    const float* y = s->output_ref->data;
    float* g       = grad_output->data;

    for (int64_t b = 0; b < batch; b++) {
        const float* yrow = y + b * classes;
        float*       grow = g + b * classes;
        float dot = 0.0f;
        for (int64_t c = 0; c < classes; c++) {
            dot += grow[c] * yrow[c];
        }
        for (int64_t c = 0; c < classes; c++) {
            grow[c] = yrow[c] * (grow[c] - dot);
        }
    }
}

/* ---- Softmax ---- */

ce_module_t* ce_softmax_create(const char* name) {
    ce_module_t* mod = ce_module_create(name ? name : "softmax");
    if (!mod) return NULL;

    softmax_layer_state_t* state =
        (softmax_layer_state_t*)calloc(1, sizeof(softmax_layer_state_t));
    if (!state) { ce_module_destroy(mod); return NULL; }
    state->act_type = CE_ACT_SOFTMAX;
    state->temperature = 1.0f;
    mod->internal_state = state;

    mod->forward  = ce_softmax_forward;
    mod->backward = ce_softmax_backward;
    return mod;
}

/* ---- Gumbel-Softmax ---- */

ce_module_t* ce_gumbel_softmax_create(float temperature, const char* name) {
    if (temperature <= 0.0f) return NULL;

    ce_module_t* mod = ce_module_create(name ? name : "gumbel_softmax");
    if (!mod) return NULL;

    softmax_layer_state_t* state =
        (softmax_layer_state_t*)calloc(1, sizeof(softmax_layer_state_t));
    if (!state) { ce_module_destroy(mod); return NULL; }
    state->act_type = CE_ACT_GUMBEL_SOFTMAX;
    state->temperature = temperature;
    mod->internal_state = state;

    mod->forward  = ce_softmax_forward;
    mod->backward = ce_softmax_backward;
    return mod;
}

ce_status_t ce_softmax_forward(ce_module_t* mod,
                                ce_tensor_t* input, ce_tensor_t* output) {
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;
    softmax_layer_state_t* state = (softmax_layer_state_t*)mod->internal_state;
    if (!state) return CE_ERR_NOT_INITIALIZED;

    ce_status_t st = ce_act(input, state->act_type, 0.0f,
                            state->temperature, output);
    if (st != CE_OK) return st;

    /* Cache output for the non-tape backward (mirrors sigmoid). */
    if (state->output_cache) ce_tensor_release(state->output_cache);
    state->output_cache = ce_tensor_clone(output);

    /* Record tape entry */
    ce_tape_t* tape = ce_get_tape();
    if (tape && tape->recording) {
        softmax_state_t* ss =
            (softmax_state_t*)calloc(1, sizeof(softmax_state_t));
        if (ss) {
            ce_tensor_retain(output);
            ss->output_ref  = output;
            ss->temperature = state->temperature;
            ce_tape_record(tape, output, softmax_tape_backward, ss, 1);
        }
    }
    return CE_OK;
}

ce_status_t ce_softmax_backward(ce_module_t* mod,
                                 ce_tensor_t* grad_output,
                                 ce_tensor_t* grad_input) {
    if (!mod || !grad_output || !grad_input) return CE_ERR_INVALID_ARG;
    softmax_layer_state_t* state = (softmax_layer_state_t*)mod->internal_state;
    if (!state || !state->output_cache) return CE_ERR_NOT_INITIALIZED;

    /* JVP over the cached forward output — the same formula the tape
     * path applies, and (for Gumbel) the same "same formula" STE. */
    return ce_get_default_backend()->activation_backward(
        grad_output, state->output_cache, CE_ACT_SOFTMAX, 0.0f, grad_input);
}
