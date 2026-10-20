#ifndef CENGINE_LAYERS_H
#define CENGINE_LAYERS_H

#include "tensor.h"
#include "compute.h"
#include "cengine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Module base — common interface for all layers
 * ================================================================ */

typedef struct ce_module {
    char name[64];
    ce_tensor_list_t* parameters;   /* owned parameter tensors */
    ce_tensor_list_t* gradients;    /* corresponding gradient accumulators */
    ce_status_t (*forward)(struct ce_module* self,
                           ce_tensor_t* input, ce_tensor_t* output);
    ce_status_t (*backward)(struct ce_module* self,
                            ce_tensor_t* grad_output, ce_tensor_t* grad_input);
    void (*zero_grad)(struct ce_module* self);
    void* internal_state;           /* layer-specific state */
} ce_module_t;

/* Module lifecycle */
ce_module_t* ce_module_create(const char* name);
void         ce_module_destroy(ce_module_t* mod);

/* Parameter management */
ce_status_t  ce_module_add_parameter(ce_module_t* mod,
                                     ce_tensor_t* param, ce_tensor_t* grad);
ce_tensor_t* ce_module_get_parameter(const ce_module_t* mod, int index);
ce_tensor_t* ce_module_get_gradient(const ce_module_t* mod, int index);
int          ce_module_param_count(const ce_module_t* mod);

/* Default implementations (usable by any layer) */
void         ce_module_default_zero_grad(ce_module_t* mod);

/* Gradient clipping for module parameters */
ce_status_t  ce_clip_module_grads(ce_module_t* mod, float max_norm);

/* ================================================================
 * Linear — fully connected layer: Y = X @ W^T + b
 * ================================================================ */

typedef struct {
    ce_tensor_t* weight;        /* [out_features, in_features] */
    ce_tensor_t* bias;          /* [out_features] */
    ce_tensor_t* grad_w;        /* gradient accumulator [out_features, in_features] */
    ce_tensor_t* grad_b;        /* gradient accumulator [out_features] */
    ce_tensor_t* input_cache;   /* saved input for backward */
    int in_features;
    int out_features;
    char name[64];
} ce_linear_t;

ce_module_t* ce_linear_create(int in_features, int out_features,
                              const char* name);
ce_status_t  ce_linear_forward(ce_module_t* mod,
                               ce_tensor_t* input, ce_tensor_t* output);
ce_status_t  ce_linear_backward(ce_module_t* mod,
                                ce_tensor_t* grad_output,
                                ce_tensor_t* grad_input);

/* ================================================================
 * Activation layers — ELU, Sigmoid, Tanh, Softmax
 * ================================================================ */

/* ELU */
ce_module_t* ce_elu_create(float alpha, const char* name);
ce_status_t  ce_elu_forward(ce_module_t* mod,
                            ce_tensor_t* input, ce_tensor_t* output);
ce_status_t  ce_elu_backward(ce_module_t* mod,
                             ce_tensor_t* grad_output,
                             ce_tensor_t* grad_input);

/* Sigmoid */
ce_module_t* ce_sigmoid_create(const char* name);
ce_status_t  ce_sigmoid_forward(ce_module_t* mod,
                                ce_tensor_t* input, ce_tensor_t* output);
ce_status_t  ce_sigmoid_backward(ce_module_t* mod,
                                 ce_tensor_t* grad_output,
                                 ce_tensor_t* grad_input);

/* Tanh */
ce_module_t* ce_tanh_create(const char* name);
ce_status_t  ce_tanh_forward(ce_module_t* mod,
                             ce_tensor_t* input, ce_tensor_t* output);
ce_status_t  ce_tanh_backward(ce_module_t* mod,
                              ce_tensor_t* grad_output,
                              ce_tensor_t* grad_input);

/* Softmax + Gumbel-Softmax (STE) */
ce_module_t* ce_softmax_create(const char* name);
ce_module_t* ce_gumbel_softmax_create(float temperature, const char* name);
ce_status_t  ce_softmax_forward(ce_module_t* mod,
                                ce_tensor_t* input, ce_tensor_t* output);
ce_status_t  ce_softmax_backward(ce_module_t* mod,
                                 ce_tensor_t* grad_output,
                                 ce_tensor_t* grad_input);

/* ================================================================
 * Sequential — ordered container of layers
 * ================================================================ */

typedef struct {
    ce_module_t** children;
    int count;
    int capacity;
} ce_sequential_t;

ce_module_t* ce_sequential_create(const char* name);
ce_status_t  ce_sequential_add(ce_module_t* seq, ce_module_t* child);
ce_status_t  ce_sequential_forward(ce_module_t* seq,
                                   ce_tensor_t* input, ce_tensor_t* output);
ce_status_t  ce_sequential_backward(ce_module_t* seq,
                                    ce_tensor_t* grad_output,
                                    ce_tensor_t* grad_input);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_LAYERS_H */
