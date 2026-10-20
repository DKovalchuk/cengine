#ifndef CENGINE_OPTIMIZER_H
#define CENGINE_OPTIMIZER_H

#include "cengine.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- AdamW Optimizer ---- */
typedef struct {
    float lr;
    float beta1, beta2, eps, weight_decay;
    ce_tensor_t** params;  /* pointers to parameters (not owned) */
    ce_tensor_t** grads;   /* pointers to gradients (not owned) */
    ce_tensor_t** m;       /* first moment per parameter (owned) */
    ce_tensor_t** v;       /* second moment per parameter (owned) */
    int n_params;
    int step;
} ce_adam_t;

ce_adam_t* ce_adam_create(float lr, float beta1, float beta2, float eps, float wd,
                           ce_tensor_t** params, ce_tensor_t** grads, int n_params);
ce_status_t ce_adam_step(ce_adam_t* opt);
void ce_adam_destroy(ce_adam_t* opt);

/* ---- SGD Optimizer ---- */
typedef struct {
    float lr, momentum, weight_decay;
    ce_tensor_t** params;
    ce_tensor_t** grads;
    ce_tensor_t** velocity;  /* momentum buffer (owned) */
    int n_params;
} ce_sgd_t;

ce_sgd_t* ce_sgd_create(float lr, float momentum, float wd,
                          ce_tensor_t** params, ce_tensor_t** grads, int n_params);
ce_status_t ce_sgd_step(ce_sgd_t* opt);
void ce_sgd_destroy(ce_sgd_t* opt);

/* ---- Learning Rate Scheduler ---- */
typedef struct {
    float max_lr, min_lr, pct_start;
    float div_factor, final_div_factor;
    int total_steps;
    int current_step;
    float current_lr;
} ce_scheduler_t;

ce_scheduler_t* ce_scheduler_create(float max_lr, float pct_start,
                                     float div_factor, float final_div_factor,
                                     int total_steps);
float ce_scheduler_step(ce_scheduler_t* sched);
void ce_scheduler_reset(ce_scheduler_t* sched);
void ce_scheduler_destroy(ce_scheduler_t* sched);

/* ---- Gradient Clipping ---- */
float ce_compute_grad_norm(ce_tensor_t** grads, int n_grads);
ce_status_t ce_clip_grad_norm(ce_tensor_t** grads, int n_grads, float max_norm);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_OPTIMIZER_H */
