/* softmax.c — Softmax and Gumbel-Softmax (STE) activation layers */
#include "layers.h"
#include "autograd.h"
#include <stdlib.h>

typedef struct {
    ce_tensor_t* output_ref;
    float temperature;       /* 0.0 for standard softmax */
} softmax_state_t;

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

    mod->forward  = ce_softmax_forward;
    mod->backward = ce_softmax_backward;
    return mod;
}

/* ---- Gumbel-Softmax ---- */

ce_module_t* ce_gumbel_softmax_create(float temperature, const char* name) {
    if (temperature <= 0.0f) return NULL;

    ce_module_t* mod = ce_module_create(name ? name : "gumbel_softmax");
    if (!mod) return NULL;

    float* temp_ptr = (float*)calloc(1, sizeof(float));
    if (!temp_ptr) { ce_module_destroy(mod); return NULL; }
    *temp_ptr = temperature;
    mod->internal_state = temp_ptr;

    mod->forward  = ce_softmax_forward;
    mod->backward = ce_softmax_backward;
    return mod;
}

ce_status_t ce_softmax_forward(ce_module_t* mod,
                                ce_tensor_t* input, ce_tensor_t* output) {
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;

    /* Determine activation type and temperature */
    ce_activation_t act_type = CE_ACT_SOFTMAX;
    float temperature = 1.0f;
    if (mod->internal_state) {
        act_type = CE_ACT_GUMBEL_SOFTMAX;
        temperature = *(float*)mod->internal_state;
    }

    ce_status_t st = ce_act(input, act_type, 0.0f, temperature, output);
    if (st != CE_OK) return st;

    /* Record tape entry */
    ce_tape_t* tape = ce_get_tape();
    if (tape && tape->recording) {
        softmax_state_t* ss =
            (softmax_state_t*)calloc(1, sizeof(softmax_state_t));
        if (ss) {
            ce_tensor_retain(output);
            ss->output_ref  = output;
            ss->temperature = temperature;
            ce_tape_record(tape, output, softmax_tape_backward, ss, 1);
        }
    }
    return CE_OK;
}

ce_status_t ce_softmax_backward(ce_module_t* mod,
                                 ce_tensor_t* grad_output,
                                 ce_tensor_t* grad_input) {
    if (!mod || !grad_output || !grad_input) return CE_ERR_INVALID_ARG;
    (void)grad_input;
    return CE_OK;
}
