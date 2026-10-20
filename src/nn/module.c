/* module.c — Base module: parameter list management, name management */
#include "layers.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>

ce_module_t* ce_module_create(const char* name) {
    ce_module_t* mod = (ce_module_t*)calloc(1, sizeof(ce_module_t));
    if (!mod) return NULL;

    if (name) {
        strncpy(mod->name, name, sizeof(mod->name) - 1);
        mod->name[sizeof(mod->name) - 1] = '\0';
    }

    mod->parameters = ce_tensor_list_create(4);
    if (!mod->parameters) { free(mod); return NULL; }

    mod->gradients = ce_tensor_list_create(4);
    if (!mod->gradients) {
        ce_tensor_list_destroy(mod->parameters);
        free(mod);
        return NULL;
    }

    mod->zero_grad = ce_module_default_zero_grad;
    return mod;
}

void ce_module_destroy(ce_module_t* mod) {
    if (!mod) return;
    if (mod->parameters) ce_tensor_list_destroy(mod->parameters);
    if (mod->gradients)  ce_tensor_list_destroy(mod->gradients);
    /* Don't free internal_state — RecurrentExpert's saved tensors
       are referenced by Python's output tensor.  Leak is acceptable
       (process exit cleans up). */
    free(mod);
}

ce_status_t ce_module_add_parameter(ce_module_t* mod,
                                     ce_tensor_t* param, ce_tensor_t* grad) {
    if (!mod || !param) return CE_ERR_INVALID_ARG;
    ce_status_t st;

    st = ce_tensor_list_append(mod->parameters, param);
    if (st != CE_OK) return st;

    if (grad) {
        st = ce_tensor_list_append(mod->gradients, grad);
        if (st != CE_OK) return st;
    } else {
        /* Append a NULL placeholder to keep indices aligned */
        ce_tensor_retain(param); /* extra retain to balance */
        ce_tensor_release(param);
        if (mod->gradients->count < mod->gradients->capacity) {
            mod->gradients->tensors[mod->gradients->count++] = NULL;
        }
    }
    return CE_OK;
}

ce_tensor_t* ce_module_get_parameter(const ce_module_t* mod, int index) {
    if (!mod || !mod->parameters) return NULL;
    if (index < 0 || index >= mod->parameters->count) return NULL;
    return mod->parameters->tensors[index];
}

ce_tensor_t* ce_module_get_gradient(const ce_module_t* mod, int index) {
    if (!mod || !mod->gradients) return NULL;
    if (index < 0 || index >= mod->gradients->count) return NULL;
    return mod->gradients->tensors[index];
}

int ce_module_param_count(const ce_module_t* mod) {
    if (!mod || !mod->parameters) return 0;
    return mod->parameters->count;
}

void ce_module_default_zero_grad(ce_module_t* mod) {
    if (!mod || !mod->gradients) return;
    for (int i = 0; i < mod->gradients->count; i++) {
        ce_tensor_t* g = mod->gradients->tensors[i];
        if (g) ce_tensor_zero(g);
    }
}
