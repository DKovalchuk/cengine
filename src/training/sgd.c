#include "optimizer.h"
#include "simd.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>

ce_sgd_t* ce_sgd_create(float lr, float momentum, float wd,
                          ce_tensor_t** params, ce_tensor_t** grads, int n_params) {
    if (!params || !grads || n_params <= 0) return NULL;

    ce_sgd_t* opt = (ce_sgd_t*)malloc(sizeof(ce_sgd_t));
    if (!opt) return NULL;

    opt->lr = lr;
    opt->momentum = momentum;
    opt->weight_decay = wd;
    opt->n_params = n_params;
    opt->params = params;
    opt->grads = grads;

    opt->velocity = (ce_tensor_t**)malloc(n_params * sizeof(ce_tensor_t*));
    if (!opt->velocity) {
        free(opt);
        return NULL;
    }

    for (int i = 0; i < n_params; i++) {
        opt->velocity[i] = ce_tensor_zeros(params[i]->shape, params[i]->ndim);
        if (!opt->velocity[i]) {
            for (int j = 0; j < i; j++) {
                ce_tensor_release(opt->velocity[j]);
            }
            free(opt->velocity);
            free(opt);
            return NULL;
        }
    }

    return opt;
}

ce_status_t ce_sgd_step(ce_sgd_t* opt) {
    if (!opt) return CE_ERR_INVALID_ARG;

    float lr = opt->lr;
    float momentum = opt->momentum;
    float wd = opt->weight_decay;

    for (int p = 0; p < opt->n_params; p++) {
        ce_tensor_t* param = opt->params[p];
        ce_tensor_t* grad = opt->grads[p];
        int64_t n = param->numel;
        float* __restrict__ p_data = param->data;
        float* __restrict__ g_data = grad->data;
        float* __restrict__ v_data = opt->velocity[p]->data;

        int64_t i = 0;
#if CE_HAS_AVX2
        __m256 vlr = _mm256_set1_ps(lr);
        __m256 vmomentum = _mm256_set1_ps(momentum);
        __m256 vwd = _mm256_set1_ps(wd);

        for (; i + 7 < n; i += 8) {
            __m256 vp = _mm256_loadu_ps(&p_data[i]);
            __m256 vg = _mm256_loadu_ps(&g_data[i]);
            __m256 vv = _mm256_loadu_ps(&v_data[i]);

            /* Apply weight decay: grad += wd * param */
            vg = _mm256_fmadd_ps(vwd, vp, vg);

            /* Update velocity: v = momentum * v + grad */
            vv = _mm256_fmadd_ps(vmomentum, vv, vg);

            /* Update parameter: param -= lr * v */
            _mm256_storeu_ps(&p_data[i], _mm256_fnmadd_ps(vlr, vv, vp));
            _mm256_storeu_ps(&v_data[i], vv);
        }
#endif
        /* Scalar tail */
        for (; i < n; i++) {
            float g = g_data[i] + wd * p_data[i];
            float v_new = momentum * v_data[i] + g;
            p_data[i] -= lr * v_new;
            v_data[i] = v_new;
        }
    }

    return CE_OK;
}

void ce_sgd_destroy(ce_sgd_t* opt) {
    if (!opt) return;
    for (int i = 0; i < opt->n_params; i++) {
        ce_tensor_release(opt->velocity[i]);
    }
    free(opt->velocity);
    free(opt);
}
