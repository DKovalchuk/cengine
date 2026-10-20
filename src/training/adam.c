#include "optimizer.h"
#include "simd.h"
#include "memory.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

ce_adam_t* ce_adam_create(float lr, float beta1, float beta2, float eps, float wd,
                           ce_tensor_t** params, ce_tensor_t** grads, int n_params) {
    if (!params || !grads || n_params <= 0) return NULL;

    ce_adam_t* opt = (ce_adam_t*)malloc(sizeof(ce_adam_t));
    if (!opt) return NULL;

    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = wd;
    opt->n_params = n_params;
    opt->step = 0;
    opt->params = params;
    opt->grads = grads;

    opt->m = (ce_tensor_t**)malloc(n_params * sizeof(ce_tensor_t*));
    opt->v = (ce_tensor_t**)malloc(n_params * sizeof(ce_tensor_t*));
    if (!opt->m || !opt->v) {
        free(opt->m);
        free(opt->v);
        free(opt);
        return NULL;
    }

    for (int i = 0; i < n_params; i++) {
        opt->m[i] = ce_tensor_zeros(params[i]->shape, params[i]->ndim);
        opt->v[i] = ce_tensor_zeros(params[i]->shape, params[i]->ndim);
        if (!opt->m[i] || !opt->v[i]) {
            for (int j = 0; j < i; j++) {
                ce_tensor_release(opt->m[j]);
                ce_tensor_release(opt->v[j]);
            }
            free(opt->m);
            free(opt->v);
            free(opt);
            return NULL;
        }
    }

    return opt;
}

ce_status_t ce_adam_step(ce_adam_t* opt) {
    if (!opt) return CE_ERR_INVALID_ARG;

    opt->step++;
    float beta1 = opt->beta1;
    float beta2 = opt->beta2;
    float eps = opt->eps;
    float wd = opt->weight_decay;
    float lr = opt->lr;

    float bias_correction1 = 1.0f - powf(beta1, (float)opt->step);
    float bias_correction2 = 1.0f - powf(beta2, (float)opt->step);
    float step_size = lr * sqrtf(bias_correction2) / bias_correction1;

    for (int p = 0; p < opt->n_params; p++) {
        ce_tensor_t* param = opt->params[p];
        ce_tensor_t* grad = opt->grads[p];
        int64_t n = param->numel;
        float* __restrict__ p_data = param->data;
        float* __restrict__ g_data = grad->data;
        float* __restrict__ m_data = opt->m[p]->data;
        float* __restrict__ v_data = opt->v[p]->data;

        int64_t i = 0;
#if CE_HAS_AVX2
        __m256 vbeta1 = _mm256_set1_ps(beta1);
        __m256 vbeta2 = _mm256_set1_ps(beta2);
        __m256 vone_minus_beta1 = _mm256_set1_ps(1.0f - beta1);
        __m256 vone_minus_beta2 = _mm256_set1_ps(1.0f - beta2);
        __m256 veps = _mm256_set1_ps(eps);
        __m256 vwd = _mm256_set1_ps(wd);
        __m256 vstep_size = _mm256_set1_ps(step_size);

        for (; i + 7 < n; i += 8) {
            __m256 vp = _mm256_loadu_ps(&p_data[i]);
            __m256 vg = _mm256_loadu_ps(&g_data[i]);
            __m256 vm = _mm256_loadu_ps(&m_data[i]);
            __m256 vv = _mm256_loadu_ps(&v_data[i]);

            /* Apply decoupled weight decay: grad += wd * param */
            vg = _mm256_fmadd_ps(vwd, vp, vg);

            /* Update first moment: m = beta1 * m + (1 - beta1) * grad */
            vm = _mm256_fmadd_ps(vbeta1, vm, _mm256_mul_ps(vone_minus_beta1, vg));

            /* Update second moment: v = beta2 * v + (1 - beta2) * grad^2 */
            vv = _mm256_fmadd_ps(vbeta2, vv, _mm256_mul_ps(vone_minus_beta2, _mm256_mul_ps(vg, vg)));

            /* Compute update: step_size * m / (sqrt(v) + eps) */
            __m256 update = _mm256_mul_ps(vstep_size,
                _mm256_div_ps(vm, _mm256_add_ps(_mm256_sqrt_ps(vv), veps)));

            /* Update parameter: param -= update */
            _mm256_storeu_ps(&p_data[i], _mm256_sub_ps(vp, update));
            _mm256_storeu_ps(&m_data[i], vm);
            _mm256_storeu_ps(&v_data[i], vv);
        }
#endif
        /* Scalar tail */
        for (; i < n; i++) {
            float g = g_data[i] + wd * p_data[i];
            float m_new = beta1 * m_data[i] + (1.0f - beta1) * g;
            float v_new = beta2 * v_data[i] + (1.0f - beta2) * g * g;
            float update = step_size * m_new / (sqrtf(v_new) + eps);
            p_data[i] -= update;
            m_data[i] = m_new;
            v_data[i] = v_new;
        }
    }

    return CE_OK;
}

void ce_adam_destroy(ce_adam_t* opt) {
    if (!opt) return;
    for (int i = 0; i < opt->n_params; i++) {
        ce_tensor_release(opt->m[i]);
        ce_tensor_release(opt->v[i]);
    }
    free(opt->m);
    free(opt->v);
    free(opt);
}
