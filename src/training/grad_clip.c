#include "optimizer.h"
#include "simd.h"
#include <math.h>

float ce_compute_grad_norm(ce_tensor_t** grads, int n_grads) {
    if (!grads || n_grads <= 0) return 0.0f;

    double total_norm_sq = 0.0;

    for (int g = 0; g < n_grads; g++) {
        ce_tensor_t* grad = grads[g];
        if (!grad || !grad->data) continue;

        int64_t n = grad->numel;
        const float* __restrict__ data = grad->data;
        int64_t i = 0;

#if CE_HAS_AVX2
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 7 < n; i += 8) {
            __m256 vg = _mm256_loadu_ps(&data[i]);
            vsum = _mm256_fmadd_ps(vg, vg, vsum);
        }
        /* Horizontal sum */
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float partial;
        _mm_store_ss(&partial, s);
        total_norm_sq += (double)partial;
#endif
        for (; i < n; i++) {
            total_norm_sq += (double)data[i] * (double)data[i];
        }
    }

    return (float)sqrt(total_norm_sq);
}

ce_status_t ce_clip_grad_norm(ce_tensor_t** grads, int n_grads, float max_norm) {
    if (!grads || n_grads <= 0) return CE_ERR_INVALID_ARG;

    float total_norm = ce_compute_grad_norm(grads, n_grads);
    if (total_norm <= max_norm) return CE_OK;

    float clip_coef = max_norm / total_norm;

    for (int g = 0; g < n_grads; g++) {
        ce_tensor_t* grad = grads[g];
        if (!grad || !grad->data) continue;

        int64_t n = grad->numel;
        float* __restrict__ data = grad->data;
        int64_t i = 0;

#if CE_HAS_AVX2
        __m256 vclip = _mm256_set1_ps(clip_coef);
        for (; i + 7 < n; i += 8) {
            __m256 vg = _mm256_loadu_ps(&data[i]);
            _mm256_storeu_ps(&data[i], _mm256_mul_ps(vg, vclip));
        }
#endif
        for (; i < n; i++) {
            data[i] *= clip_coef;
        }
    }

    return CE_OK;
}
