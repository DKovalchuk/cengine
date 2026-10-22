/* elu.c — ELU activation layer with cached output for backward */
#include "layers.h"
#include "autograd.h"
#include <stdlib.h>
#include <string.h>

/* Layer state: alpha + cached output for backward */
typedef struct {
    float alpha;
    ce_tensor_t* output_cache;
} elu_layer_state_t;

/* State for tape backward */
typedef struct {
    ce_tensor_t* output_ref;
    float alpha;
} elu_state_t;

static void elu_state_free(void* state) {
    elu_state_t* s = (elu_state_t*)state;
    if (s->output_ref) ce_tensor_release(s->output_ref);
    free(s);
}

static void elu_tape_backward(void* state, ce_tensor_t* grad_output) {
    elu_state_t* s = (elu_state_t*)state;
    if (!s || !grad_output || !s->output_ref) return;
    ce_get_default_backend()->activation_backward(
        grad_output, s->output_ref, CE_ACT_ELU, s->alpha, grad_output);
}

ce_module_t* ce_elu_create(float alpha, const char* name) {
    if (alpha <= 0.0f) return NULL;
    ce_module_t* mod = ce_module_create(name ? name : "elu");
    if (!mod) return NULL;

    elu_layer_state_t* state = (elu_layer_state_t*)calloc(1, sizeof(elu_layer_state_t));
    if (!state) { ce_module_destroy(mod); return NULL; }
    state->alpha = alpha;
    state->output_cache = NULL;
    mod->internal_state = state;
    mod->forward  = ce_elu_forward;
    mod->backward = ce_elu_backward;
    return mod;
}

ce_status_t ce_elu_forward(ce_module_t* mod,
                            ce_tensor_t* input, ce_tensor_t* output) {
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;
    elu_layer_state_t* state = (elu_layer_state_t*)mod->internal_state;
    if (!state) return CE_ERR_NOT_INITIALIZED;

    ce_status_t st = ce_act(input, CE_ACT_ELU, state->alpha, 0.0f, output);
    if (st != CE_OK) return st;

    /* Cache output for backward */
    if (state->output_cache) ce_tensor_release(state->output_cache);
    state->output_cache = ce_tensor_clone(output);

    /* Record tape entry */
    ce_tape_t* tape = ce_get_tape();
    if (tape && tape->recording) {
        elu_state_t* es = (elu_state_t*)calloc(1, sizeof(elu_state_t));
        if (es) {
            ce_tensor_retain(state->output_cache);
            es->output_ref = state->output_cache;
            es->alpha = state->alpha;
            ce_tape_record(tape, output, elu_tape_backward, es, 1,
                           elu_state_free);
        }
    }
    return CE_OK;
}

ce_status_t ce_elu_backward(ce_module_t* mod,
                             ce_tensor_t* grad_output,
                             ce_tensor_t* grad_input) {
    if (!mod || !grad_output || !grad_input) return CE_ERR_INVALID_ARG;
    elu_layer_state_t* state = (elu_layer_state_t*)mod->internal_state;
    if (!state || !state->output_cache) return CE_ERR_NOT_INITIALIZED;

    /* ELU backward: grad_input = grad_output * d/dx(ELU(x))
     * d/dx ELU(x) = 1 if x > 0, else ELU(x) + alpha = output + alpha */
    return ce_get_default_backend()->activation_backward(
        grad_output, state->output_cache, CE_ACT_ELU, state->alpha, grad_input);
}
