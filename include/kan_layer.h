#ifndef CENGINE_KAN_LAYER_H
#define CENGINE_KAN_LAYER_H

#include "layers.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * KAN Layer — Polynomial Basis + SiLU base
 *
 * y_j = sum_i (w_b[j,i] * SiLU(x_i) + w_s[j,i] * sum_k c[j,i,k] * P_k(x_i))
 *     + bias[j]
 *
 * Polynomial basis: P(x) = [1, x, x^2, x^3]
 * Analytical gradient: dP/dx = [0, 1, 2x, 3x^2]
 *
 * Parameters:
 *   w_b:   [out_features, in_features] — base activation weights
 *   w_s:   [out_features, in_features] — spline weights
 *   c:     [out_features, in_features, 4] — polynomial coefficients
 *   bias:  [out_features]
 *
 * Input:  [batch, seq_len, in_features]
 * Output: [batch, seq_len, out_features]
 * ================================================================ */

typedef struct {
    /* Parameters */
    ce_tensor_t* w_b;        /* [out_f, in_f] */
    ce_tensor_t* w_s;        /* [out_f, in_f] */
    ce_tensor_t* c;          /* [out_f, in_f, 4] */
    ce_tensor_t* bias;       /* [out_f] */

    /* Gradient accumulators */
    ce_tensor_t* g_w_b;
    ce_tensor_t* g_w_s;
    ce_tensor_t* g_c;
    ce_tensor_t* g_bias;

    /* Saved state for backward */
    ce_tensor_t* saved_basis;    /* [N, in_f, 4] */
    ce_tensor_t* saved_silu;     /* [N, in_f] */
    ce_tensor_t* saved_x_norm;   /* [B, T, in_f] */
    ce_tensor_t* saved_c64;      /* [out_f, in_f, 4] */

    int in_features;
    int out_features;
    char name[64];
} ce_kan_layer_t;

ce_module_t* ce_kan_layer_create(int in_features, int out_features,
                                  const char* name);
ce_status_t  ce_kan_layer_forward(ce_module_t* mod,
                                   ce_tensor_t* input,
                                   ce_tensor_t* output);
ce_status_t  ce_kan_layer_backward(ce_module_t* mod,
                                    ce_tensor_t* grad_output,
                                    ce_tensor_t* grad_input);

/* Batch forward: process [B, T, in_f] → [B, T, out_f] */
ce_status_t  ce_kan_layer_forward_batch(ce_module_t* mod,
                                         ce_tensor_t* input,
                                         ce_tensor_t* output);

/* Batch backward */
ce_status_t  ce_kan_layer_backward_batch(ce_module_t* mod,
                                          ce_tensor_t* grad_output,
                                          ce_tensor_t* grad_input);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_KAN_LAYER_H */
