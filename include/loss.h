#ifndef CENGINE_LOSS_H
#define CENGINE_LOSS_H

#include "cengine.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Binary Cross-Entropy with Logits ---- */
float ce_bce_with_logits_loss(const ce_tensor_t* logits, const ce_tensor_t* targets,
                               const ce_tensor_t* mask);
ce_status_t ce_bce_with_logits_backward(const ce_tensor_t* logits, const ce_tensor_t* targets,
                                         const ce_tensor_t* mask, ce_tensor_t* grad);

/* ---- Mean Squared Error ----
 * Optional mask (mirrors BCE): NULL → mean over all elements (legacy
 * behaviour); otherwise zero entries are excluded from the mean and get
 * zero gradient, and the divisor is the number of valid entries. */
float ce_mse_loss(const ce_tensor_t* pred, const ce_tensor_t* targets,
                  const ce_tensor_t* mask);
ce_status_t ce_mse_backward(const ce_tensor_t* pred, const ce_tensor_t* targets,
                            const ce_tensor_t* mask, ce_tensor_t* grad);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_LOSS_H */
