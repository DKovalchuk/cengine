#include "loss.h"
#include "simd.h"

/*
 * Masked semantics (mirrors ce_bce_with_logits_loss/backward):
 *   mask == NULL        → mean over all n elements (legacy behaviour);
 *   mask entries == 0   → excluded from the loss sum, zero gradient,
 *                          divisor = number of valid entries.
 */
float ce_mse_loss(const ce_tensor_t* pred, const ce_tensor_t* targets,
                  const ce_tensor_t* mask) {
    if (!pred || !targets) return 0.0f;
    if (pred->numel != targets->numel) return 0.0f;
    if (mask && mask->numel != pred->numel) return 0.0f;

    int64_t n = pred->numel;
    const float* __restrict__ p_data = pred->data;
    const float* __restrict__ t_data = targets->data;
    const float* __restrict__ m_data = mask ? mask->data : NULL;
    double loss_sum = 0.0;
    double n_valid = 0.0;

    if (!m_data) {
        /* Unmasked fast path — same as the original implementation. */
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

    /* Masked path: accumulate only valid entries, mean over n_valid. */
    for (int64_t i = 0; i < n; i++) {
        if (m_data[i] == 0.0f) continue;
        double diff = (double)p_data[i] - (double)t_data[i];
        loss_sum += diff * diff;
        n_valid += 1.0;
    }

    if (n_valid > 0.0) {
        return (float)(loss_sum / n_valid);
    }
    return 0.0f;
}

ce_status_t ce_mse_backward(const ce_tensor_t* pred, const ce_tensor_t* targets,
                            const ce_tensor_t* mask, ce_tensor_t* grad) {
    if (!pred || !targets || !grad) return CE_ERR_INVALID_ARG;
    if (pred->numel != targets->numel || pred->numel != grad->numel) return CE_ERR_INVALID_SHAPE;
    if (mask && mask->numel != pred->numel) return CE_ERR_INVALID_SHAPE;

    int64_t n = pred->numel;
    const float* __restrict__ p_data = pred->data;
    const float* __restrict__ t_data = targets->data;
    const float* __restrict__ m_data = mask ? mask->data : NULL;
    float* __restrict__ g_data = grad->data;

    if (!m_data) {
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

    /* Masked path: divisor is n_valid, padded entries get zero grad. */
    double n_valid = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (m_data[i] != 0.0f) n_valid += 1.0;
    }
    if (n_valid <= 0.0) {
        for (int64_t i = 0; i < n; i++) g_data[i] = 0.0f;
        return CE_OK;
    }

    float scale = 2.0f / (float)n_valid;
    int64_t i = 0;
#if CE_HAS_AVX2
    __m256 vscale = _mm256_set1_ps(scale);
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 vp = _mm256_loadu_ps(&p_data[i]);
        __m256 vt = _mm256_loadu_ps(&t_data[i]);
        __m256 vm = _mm256_loadu_ps(&m_data[i]);
        __m256 diff = _mm256_sub_ps(vp, vt);
        __m256 vg = _mm256_mul_ps(diff, vscale);
        /* Zero out padded entries (mask == 0). */
        __m256 mask_cmp = _mm256_cmp_ps(vm, vzero, _CMP_NEQ_OQ);
        vg = _mm256_and_ps(vg, mask_cmp);
        _mm256_storeu_ps(&g_data[i], vg);
    }
#endif
    for (; i < n; i++) {
        if (m_data[i] == 0.0f) {
            g_data[i] = 0.0f;
            continue;
        }
        g_data[i] = (p_data[i] - t_data[i]) * scale;
    }

    return CE_OK;
}
