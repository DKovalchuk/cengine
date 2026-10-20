#ifndef CENGINE_COMPUTE_H
#define CENGINE_COMPUTE_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct ce_compute_backend;

/* Matmul flags */
typedef enum {
    CE_TRANSPOSE_NONE = 0,
    CE_TRANSPOSE_A    = 1,
    CE_TRANSPOSE_B    = 2,
} ce_transpose_t;

/* Activation types */
typedef enum {
    CE_ACT_RELU = 0,
    CE_ACT_ELU,
    CE_ACT_SIGMOID,
    CE_ACT_TANH,
    CE_ACT_SOFTMAX,
    CE_ACT_GUMBEL_SOFTMAX,
} ce_activation_t;

/* Backend operations vtable */
typedef struct ce_compute_backend {
    const char* name;
    ce_device_t device;

    /* Math */
    ce_status_t (*matmul)(const ce_tensor_t* a, const ce_tensor_t* b,
                          ce_tensor_t* out, ce_transpose_t trans);
    ce_status_t (*add)(const ce_tensor_t* a, const ce_tensor_t* b, ce_tensor_t* out);
    ce_status_t (*mul)(const ce_tensor_t* a, const ce_tensor_t* b, ce_tensor_t* out);
    ce_status_t (*axpy)(float alpha, const ce_tensor_t* x, ce_tensor_t* y);
    ce_status_t (*activation)(const ce_tensor_t* in, ce_activation_t type,
                              float alpha, float temperature, ce_tensor_t* out);
    ce_status_t (*activation_backward)(const ce_tensor_t* grad_out,
                                       const ce_tensor_t* act_out,
                                       ce_activation_t type, float alpha,
                                       ce_tensor_t* grad_in);

    /* Reductions */
    ce_status_t (*sum)(const ce_tensor_t* in, ce_tensor_t* out, int dim);
    ce_status_t (*mean)(const ce_tensor_t* in, ce_tensor_t* out, int dim);
    ce_status_t (*max)(const ce_tensor_t* in, ce_tensor_t* out, int dim, ce_tensor_t* indices);
    float       (*norm)(const ce_tensor_t* in);
    float       (*dot)(const ce_tensor_t* a, const ce_tensor_t* b);

    /* Info */
    int (*get_max_threads)(void);
} ce_compute_backend_t;

/* Backend registry */
ce_status_t             ce_register_backend(ce_compute_backend_t* backend);
ce_compute_backend_t*   ce_get_backend(ce_device_t device);
ce_compute_backend_t*   ce_get_default_backend(void);

/* Convenience macros */
#define ce_matmul(a, b, out, trans) ce_get_default_backend()->matmul(a, b, out, trans)
#define ce_add(a, b, out)           ce_get_default_backend()->add(a, b, out)
#define ce_mul(a, b, out)           ce_get_default_backend()->mul(a, b, out)
#define ce_axpy(alpha, x, y)        ce_get_default_backend()->axpy(alpha, x, y)
#define ce_act(in, type, alpha, temp, out) ce_get_default_backend()->activation(in, type, alpha, temp, out)
#define ce_act_bw(go, ao, type, alpha, gi) ce_get_default_backend()->activation_backward(go, ao, type, alpha, gi)

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_COMPUTE_H */
