/* sigmoid.c — Sigmoid activation layer with cached output for backward */
#include "layers.h"
#include "autograd.h"
#include <stdlib.h>

typedef struct {
    ce_tensor_t* output_cache;
} sigmoid_layer_state_t;

typedef struct {
    ce_tensor_t* output_ref;
} sigmoid_state_t;

static void sigmoid_state_free(void* state) {
    sigmoid_state_t* s = (sigmoid_state_t*)state;
    if (s->output_ref) ce_tensor_release(s->output_ref);
    free(s);
}

static void sigmoid_tape_backward(void* state, ce_tensor_t* grad_output) {
    sigmoid_state_t* s = (sigmoid_state_t*)state;
    if (!s || !grad_output || !s->output_ref) return;
    ce_get_default_backend()->activation_backward(
        grad_output, s->output_ref, CE_ACT_SIGMOID, 0.0f, grad_output);
}

ce_module_t* ce_sigmoid_create(const char* name) {
    ce_module_t* mod = ce_module_create(name ? name : "sigmoid");
    if (!mod) return NULL;

    sigmoid_layer_state_t* state = (sigmoid_layer_state_t*)calloc(1, sizeof(sigmoid_layer_state_t));
    if (!state) { ce_module_destroy(mod); return NULL; }
    mod->internal_state = state;
    mod->forward  = ce_sigmoid_forward;
    mod->backward = ce_sigmoid_backward;
    return mod;
}

ce_status_t ce_sigmoid_forward(ce_module_t* mod,
                                ce_tensor_t* input, ce_tensor_t* output) {
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;
    sigmoid_layer_state_t* state = (sigmoid_layer_state_t*)mod->internal_state;
    if (!state) return CE_ERR_NOT_INITIALIZED;

    ce_status_t st = ce_act(input, CE_ACT_SIGMOID, 0.0f, 0.0f, output);
    if (st != CE_OK) return st;

    if (state->output_cache) ce_tensor_release(state->output_cache);
    state->output_cache = ce_tensor_clone(output);

    ce_tape_t* tape = ce_get_tape();
    if (tape && tape->recording) {
        sigmoid_state_t* ss = (sigmoid_state_t*)calloc(1, sizeof(sigmoid_state_t));
        if (ss) {
            ce_tensor_retain(state->output_cache);
            ss->output_ref = state->output_cache;
            ce_tape_record(tape, output, sigmoid_tape_backward, ss, 1,
                           sigmoid_state_free);
        }
    }
    return CE_OK;
}

ce_status_t ce_sigmoid_backward(ce_module_t* mod,
                                 ce_tensor_t* grad_output,
                                 ce_tensor_t* grad_input) {
    if (!mod || !grad_output || !grad_input) return CE_ERR_INVALID_ARG;
    sigmoid_layer_state_t* state = (sigmoid_layer_state_t*)mod->internal_state;
    if (!state || !state->output_cache) return CE_ERR_NOT_INITIALIZED;

    /* Sigmoid backward: grad = grad_output * sigmoid(x) * (1 - sigmoid(x))
     * = grad_output * output * (1 - output) */
    return ce_get_default_backend()->activation_backward(
        grad_output, state->output_cache, CE_ACT_SIGMOID, 0.0f, grad_input);
}
