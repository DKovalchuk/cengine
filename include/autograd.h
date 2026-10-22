#ifndef CENGINE_AUTOGRAD_H
#define CENGINE_AUTOGRAD_H

#include "tensor.h"
#include "cengine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Tape entry — one recorded operation in the backward graph
 * ================================================================ */

typedef struct {
    ce_tensor_t* output;                 /* output tensor of this op */
    ce_tensor_t* grad;                   /* accumulated gradient */
    void (*backward_fn)(void* state,     /* layer-specific backward */
                        ce_tensor_t* grad_output);
    void* state;                         /* layer-specific state (owned) */
    void (*state_free)(void* state);     /* releases state's retains */
    int n_inputs;                        /* number of gradient inputs */
} tape_entry_t;

/* ================================================================
 * Gradient tape — records ops during forward, replays in reverse
 * ================================================================ */

#define CE_TAPE_INITIAL_CAP 64

typedef struct {
    tape_entry_t* entries;
    int length;
    int capacity;
    bool recording;
} ce_tape_t;

/* Tape lifecycle */
ce_tape_t*   ce_tape_create(void);
void         ce_tape_destroy(ce_tape_t* tape);

/* Recording.  `state_free` releases whatever `state` retained before
 * `state` itself is freed — without it the tape leaked every tensor a
 * recorded state kept (retained input/output caches, weights). */
ce_status_t  ce_tape_start(ce_tape_t* tape);
ce_status_t  ce_tape_stop(ce_tape_t* tape);
ce_status_t  ce_tape_record(ce_tape_t* tape,
                            ce_tensor_t* output,
                            void (*backward_fn)(void*, ce_tensor_t*),
                            void* state, int n_inputs,
                            void (*state_free)(void*));

/* Backward */
ce_status_t  ce_tape_backward(ce_tape_t* tape,
                              ce_tensor_t* loss_grad);

/* Global tape access */
ce_tape_t*   ce_get_tape(void);
void         ce_set_tape(ce_tape_t* tape);

/* ================================================================
 * Backward engine — traverse tape and compute gradients
 * ================================================================ */

/* Run backward pass on the global tape (shorthand for ce_tape_backward) */
ce_status_t  ce_backward(ce_tensor_t* loss_grad);

/* Compute gradient of scalar loss w.r.t. registered parameters */
ce_status_t  ce_compute_grad(ce_tensor_t* loss_grad,
                             ce_tensor_t* target_grad_out);

/* Gradient clipping — use ce_clip_grad_norm from training/grad_clip.c
 * and ce_clip_module_grads from layers.h */

/* ================================================================
 * Graph node helpers (for advanced/manual graph construction)
 * ================================================================ */

tape_entry_t* ce_graph_node_create(ce_tensor_t* output,
                                    void (*backward_fn)(void*, ce_tensor_t*),
                                    void* state, int n_inputs);
void          ce_graph_node_destroy(tape_entry_t* node);
ce_status_t   ce_graph_node_ensure_grad(tape_entry_t* node);
ce_tensor_t*  ce_graph_node_get_grad(const tape_entry_t* node);
ce_tensor_t*  ce_graph_node_get_output(const tape_entry_t* node);
int           ce_graph_count_dependencies(const tape_entry_t* node);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_AUTOGRAD_H */
