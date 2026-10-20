#include "loss.h"
#include "simd.h"

float ce_mse_loss(const ce_tensor_t* pred, const ce_tensor_t* targets) {
    if (!pred || !targets) return 0.0f;
    if (pred->numel != targets->numel) return 0.0f;

    int64_t n = pred->numel;
    const float* __restrict__ p_data = pred->data;
    const float* __restrict__ t_data = targets->data;
    double loss_sum = 0.0;

    int64_t i = 0;
#if CE_HAS_AVX2
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 vp = _mm256_loadu_ps(&p_data[i]);
        __m256 vt = _mm256_loadu_ps(&t_data[i]);
        __m256 diff = _mm256_sub_ps(vp, vt);
        vsum = _mm256_fmadd_ps(diff, diff, vsum);
    }
    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float partial;
    _mm_store_ss(&partial, s);
    loss_sum = (double)partial;
#endif
    for (; i < n; i++) {
        double diff = (double)p_data[i] - (double)t_data[i];
        loss_sum += diff * diff;
    }

    return (float)(loss_sum / (double)n);
}

ce_status_t ce_mse_backward(const ce_tensor_t* pred, const ce_tensor_t* targets,
                             ce_tensor_t* grad) {
    if (!pred || !targets || !grad) return CE_ERR_INVALID_ARG;
    if (pred->numel != targets->numel || pred->numel != grad->numel) return CE_ERR_INVALID_SHAPE;

    int64_t n = pred->numel;
    const float* __restrict__ p_data = pred->data;
    const float* __restrict__ t_data = targets->data;
    float* __restrict__ g_data = grad->data;
    float scale = 2.0f / (float)n;

    int64_t i = 0;
#if CE_HAS_AVX2
    __m256 vscale = _mm256_set1_ps(scale);
    for (; i + 7 < n; i += 8) {
        __m256 vp = _mm256_loadu_ps(&p_data[i]);
        __m256 vt = _mm256_loadu_ps(&t_data[i]);
        __m256 diff = _mm256_sub_ps(vp, vt);
        _mm256_storeu_ps(&g_data[i], _mm256_mul_ps(diff, vscale));
    }
#endif
    for (; i < n; i++) {
        g_data[i] = (p_data[i] - t_data[i]) * scale;
    }

    return CE_OK;
}
