/* sequential.c — Sequential container: chains layers in order */
#include "layers.h"
#include "autograd.h"
#include <stdlib.h>
#include <string.h>

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
            /* Allocate intermediate buffer.
               For layers that change dimensionality (Linear), we need
               the output shape.  Use a heuristic: if the layer name
               contains "linear" and has internal_state, try to read
               out_features from ce_linear_t.  Otherwise use same shape. */
            int64_t inter_shape[CE_MAX_DIMS];
            int inter_ndim = current->ndim;
            memcpy(inter_shape, current->shape, sizeof(int64_t) * (size_t)current->ndim);

            /* Safe heuristic: check if this layer's forward is ce_linear_forward */
            if (layer->internal_state && layer->forward == ce_linear_forward) {
                ce_linear_t* lin = (ce_linear_t*)layer->internal_state;
                if (lin->out_features > 0 && lin->in_features > 0) {
                    inter_shape[inter_ndim - 1] = lin->out_features;
                }
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

        /* Allocate intermediate gradient buffer with correct shape */
        int64_t gi_shape[CE_MAX_DIMS];
        int gi_ndim = current_grad->ndim;
        memcpy(gi_shape, current_grad->shape,
               sizeof(int64_t) * (size_t)current_grad->ndim);

        if (layer->internal_state && layer->forward == ce_linear_forward) {
            ce_linear_t* lin = (ce_linear_t*)layer->internal_state;
            if (lin->in_features > 0) {
                gi_shape[gi_ndim - 1] = lin->in_features;
            }
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
