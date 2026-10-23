/* sequential.c — Sequential container: chains layers in order */
#include "layers.h"
#include "autograd.h"
#include "kan_layer.h"
#include <stdlib.h>
#include <string.h>

/* Children that change the last dimension expose their widths through
 * their layer struct.  The intermediate/gradient buffer must follow
 * those widths: keyed only on ce_linear_forward, a KAN child mid-chain
 * got an input-sized buffer and its forward wrote out_features values
 * per row past the end (heap OOB write whenever out > in). */
static bool child_widths(const ce_module_t* layer, int* in_f, int* out_f) {
    if (!layer->internal_state) return false;
    if (layer->forward == ce_linear_forward) {
        const ce_linear_t* lin = (const ce_linear_t*)layer->internal_state;
        if (lin->in_features > 0 && lin->out_features > 0) {
            *in_f = lin->in_features;
            *out_f = lin->out_features;
            return true;
        }
    } else if (layer->forward == ce_kan_layer_forward) {
        const ce_kan_layer_t* k =
            (const ce_kan_layer_t*)layer->internal_state;
        if (k->in_features > 0 && k->out_features > 0) {
            *in_f = k->in_features;
            *out_f = k->out_features;
            return true;
        }
    }
    return false;
}

static void sequential_zero_grad(ce_module_t* mod) {
    if (!mod || !mod->internal_state) return;
    ce_sequential_t* seq = (ce_sequential_t*)mod->internal_state;
    for (int i = 0; i < seq->count; i++) {
        if (seq->children[i] && seq->children[i]->zero_grad) {
            seq->children[i]->zero_grad(seq->children[i]);
        }
    }
    /* Also zero own accumulated parameters (if any) */
    ce_module_default_zero_grad(mod);
}

ce_module_t* ce_sequential_create(const char* name) {
    ce_module_t* mod = ce_module_create(name ? name : "sequential");
    if (!mod) return NULL;

    ce_sequential_t* seq =
        (ce_sequential_t*)calloc(1, sizeof(ce_sequential_t));
    if (!seq) { ce_module_destroy(mod); return NULL; }

    seq->capacity = 8;
    seq->children = (ce_module_t**)calloc(
        (size_t)seq->capacity, sizeof(ce_module_t*));
    if (!seq->children) {
        free(seq);
        ce_module_destroy(mod);
        return NULL;
    }

    mod->internal_state = seq;
    mod->forward  = ce_sequential_forward;
    mod->backward = ce_sequential_backward;
    mod->zero_grad = sequential_zero_grad;
    return mod;
}

ce_status_t ce_sequential_add(ce_module_t* seq_mod, ce_module_t* child) {
    if (!seq_mod || !child || !seq_mod->internal_state)
        return CE_ERR_INVALID_ARG;

    ce_sequential_t* seq = (ce_sequential_t*)seq_mod->internal_state;

    /* Grow if needed */
    if (seq->count >= seq->capacity) {
        int new_cap = seq->capacity * 2;
        ce_module_t** new_arr = (ce_module_t**)realloc(
            seq->children, (size_t)new_cap * sizeof(ce_module_t*));
        if (!new_arr) return CE_ERR_OUT_OF_MEMORY;
        seq->children = new_arr;
        seq->capacity = new_cap;
    }

    /* Expose the child's parameters/gradients on the container: generic
     * consumers of ce_module_param_count (checkpoint save/load, the C
     * trainer) must see the whole model instead of an empty parent list —
     * otherwise a Sequential checkpoint is written with n_params = 0 (a
     * 20-byte header, no weights) while save/load still report CE_OK.
     * ce_tensor_list_append retains and the parent lists are destroyed
     * after the children, so ownership stays balanced.  Python's
     * Module._collect_params reads parent lists only for leaf modules,
     * so container consumers never see duplicates. */
    for (int i = 0; i < child->parameters->count; i++) {
        ce_tensor_t* p = child->parameters->tensors[i];
        if (!p) continue;
        ce_status_t st = ce_tensor_list_append(seq_mod->parameters, p);
        if (st != CE_OK) return st;
    }
    for (int i = 0; i < child->gradients->count; i++) {
        ce_tensor_t* g = child->gradients->tensors[i];
        if (!g) continue;
        ce_status_t st = ce_tensor_list_append(seq_mod->gradients, g);
        if (st != CE_OK) return st;
    }

    seq->children[seq->count++] = child;
    return CE_OK;
}

ce_status_t ce_sequential_forward(ce_module_t* seq_mod,
                                   ce_tensor_t* input, ce_tensor_t* output) {
    if (!seq_mod || !input || !output) return CE_ERR_INVALID_ARG;
    ce_sequential_t* seq = (ce_sequential_t*)seq_mod->internal_state;
    if (!seq || seq->count == 0) return CE_ERR_NOT_INITIALIZED;

    ce_status_t st;
    ce_tensor_t* current = input;
    bool allocated_intermediate = false;

    for (int i = 0; i < seq->count; i++) {
        ce_module_t* layer = seq->children[i];
        if (!layer || !layer->forward) return CE_ERR_NOT_INITIALIZED;

        if (i == seq->count - 1) {
            /* Last layer writes directly to output */
            st = layer->forward(layer, current, output);
        } else {
            /* Allocate intermediate buffer with the layer's OUTPUT
             * width when it changes dimensionality. */
            int64_t inter_shape[CE_MAX_DIMS];
            int inter_ndim = current->ndim;
            memcpy(inter_shape, current->shape, sizeof(int64_t) * (size_t)current->ndim);

            int child_in = 0, child_out = 0;
            if (child_widths(layer, &child_in, &child_out)) {
                inter_shape[inter_ndim - 1] = child_out;
            }

            ce_tensor_t* inter = ce_tensor_create(
                inter_shape, inter_ndim, current->dtype);
            if (!inter) return CE_ERR_OUT_OF_MEMORY;

            st = layer->forward(layer, current, inter);
            if (st != CE_OK) {
                ce_tensor_release(inter);
                return st;
            }

            if (allocated_intermediate) ce_tensor_release(current);
            current = inter;
            allocated_intermediate = true;
        }

        if (st != CE_OK) {
            if (allocated_intermediate) ce_tensor_release(current);
            return st;
        }
    }

    if (allocated_intermediate) ce_tensor_release(current);
    return CE_OK;
}

ce_status_t ce_sequential_backward(ce_module_t* seq_mod,
                                    ce_tensor_t* grad_output,
                                    ce_tensor_t* grad_input) {
    if (!seq_mod || !grad_output) return CE_ERR_INVALID_ARG;
    ce_sequential_t* seq = (ce_sequential_t*)seq_mod->internal_state;
    if (!seq || seq->count == 0) return CE_ERR_NOT_INITIALIZED;

    ce_status_t st;
    ce_tensor_t* current_grad = grad_output;

    for (int i = seq->count - 1; i >= 0; i--) {
        ce_module_t* layer = seq->children[i];
        if (!layer || !layer->backward) return CE_ERR_NOT_INITIALIZED;

        /* Intermediate gradient buffer follows the layer's INPUT width */
        int64_t gi_shape[CE_MAX_DIMS];
        int gi_ndim = current_grad->ndim;
        memcpy(gi_shape, current_grad->shape,
               sizeof(int64_t) * (size_t)current_grad->ndim);

        int child_in = 0, child_out = 0;
        if (child_widths(layer, &child_in, &child_out)) {
            gi_shape[gi_ndim - 1] = child_in;
        }

        ce_tensor_t* gi = ce_tensor_create(
            gi_shape, gi_ndim, current_grad->dtype);
        if (!gi) return CE_ERR_OUT_OF_MEMORY;

        st = layer->backward(layer, current_grad, gi);
        if (st != CE_OK) { ce_tensor_release(gi); return st; }

        /* Release previous intermediate gradient (not grad_output) */
        if (current_grad != grad_output) {
            ce_tensor_release(current_grad);
        }
        current_grad = gi;
    }

    /* Copy final gradient to grad_input if provided */
    if (grad_input) {
        ce_status_t cst = ce_tensor_copy(grad_input, current_grad);
        if (cst != CE_OK) return cst;
    }

    /* Release the last intermediate gradient */
    if (current_grad != grad_output) {
        ce_tensor_release(current_grad);
    }

    return CE_OK;
}

void ce_sequential_destroy(ce_module_t* mod) {
    if (!mod) return;
    ce_sequential_t* seq = (ce_sequential_t*)mod->internal_state;
    if (seq) {
        for (int i = 0; i < seq->count; i++) {
            ce_module_t* child = seq->children[i];
            if (!child) continue;
            /* Nested containers own THEIR children too: dispatch so the
             * whole tree is freed — generic ce_module_destroy knows nothing
             * about container state and would leak the subtree. */
            if (child->forward == ce_sequential_forward) {
                ce_sequential_destroy(child);
            } else {
                ce_module_destroy(child);
            }
        }
        free(seq->children);
        mod->internal_state = NULL;
        free(seq);
    }
    ce_module_destroy(mod);
}
