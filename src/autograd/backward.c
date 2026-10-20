/* backward.c — Backpropagation engine: traverse tape,
 *              call each node's backward_fn in reverse order */
#include "autograd.h"
#include "layers.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>

/* ---- Standalone backward pass ---- */

ce_status_t ce_backward(ce_tensor_t* loss_grad) {
    ce_tape_t* tape = ce_get_tape();
    if (!tape) return CE_ERR_NOT_INITIALIZED;
    return ce_tape_backward(tape, loss_grad);
}

/* ---- Gradient computation helpers ---- */

/* Compute the gradient of a scalar loss w.r.t. a specific tensor
 * by running backward on the current tape and returning the
 * gradient that was accumulated into the given grad tensor. */
ce_status_t ce_compute_grad(ce_tensor_t* loss_grad,
                             ce_tensor_t* target_grad_out) {
    if (!loss_grad || !target_grad_out) return CE_ERR_INVALID_ARG;

    ce_status_t st = ce_backward(loss_grad);
    if (st != CE_OK) return st;

    /* The gradient has been accumulated into the gradient tensors
     * registered with each module.  If the caller wants a copy
     * into target_grad_out, they should use ce_tensor_copy after
     * this returns. */
    (void)target_grad_out;
    return CE_OK;
}

/* ---- Gradient clipping utility ---- */

/* Clip gradients for all parameters of a module.
 * Delegates to ce_clip_grad_norm from training/grad_clip.c. */
extern ce_status_t ce_clip_grad_norm(ce_tensor_t** grads,
                                      int n_grads, float max_norm);

ce_status_t ce_clip_module_grads(ce_module_t* mod, float max_norm) {
    if (!mod || !mod->gradients) return CE_ERR_INVALID_ARG;
    if (mod->gradients->count <= 0) return CE_OK;
    return ce_clip_grad_norm(mod->gradients->tensors,
                              mod->gradients->count, max_norm);
}
