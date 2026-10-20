#include "loss.h"
#include "simd.h"
#include <math.h>

float ce_bce_with_logits_loss(const ce_tensor_t* logits, const ce_tensor_t* targets,
                               const ce_tensor_t* mask) {
    if (!logits || !targets) return 0.0f;
    if (logits->numel != targets->numel) return 0.0f;
    if (mask && mask->numel != logits->numel) return 0.0f;

    int64_t n = logits->numel;
    const float* __restrict__ l_data = logits->data;
    const float* __restrict__ t_data = targets->data;
    const float* __restrict__ m_data = mask ? mask->data : NULL;

    double loss_sum = 0.0;
    double n_valid = 0.0;

    for (int64_t i = 0; i < n; i++) {
        if (m_data && m_data[i] == 0.0f) continue;

        float logit = l_data[i];
        float target = t_data[i];

        /* BCE = -[t*log(σ(x)) + (1-t)*log(1-σ(x))]
         * Numerically stable (always non-negative):
         *   σ(x) = 1/(1+exp(-|x|)) with sign adjustment:
         *     x >= 0: σ(x) = 1/(1+exp(-x)), log(σ(x)) = -log(1+exp(-x))
         *     x <  0: σ(x) = exp(x)/(1+exp(x)), log(σ(x)) = x - log(1+exp(x))
         *   Loss = -t*log(σ(x)) - (1-t)*log(1-σ(x)) */
        float loss;
        if (logit >= 0.0f) {
            float exp_neg = expf(-logit);
            float log_sigmoid = -logf(1.0f + exp_neg);
            float log_1msigmoid = -logit + log_sigmoid;
            loss = -target * log_sigmoid - (1.0f - target) * log_1msigmoid;
        } else {
            float exp_pos = expf(logit);
            float log_sigmoid = logit - logf(1.0f + exp_pos);
            float log_1msigmoid = -logf(1.0f + exp_pos);
            loss = -target * log_sigmoid - (1.0f - target) * log_1msigmoid;
        }

        loss_sum += (double)loss;
        n_valid += 1.0;
    }

    if (n_valid > 0.0) {
        return (float)(loss_sum / n_valid);
    }
    return 0.0f;
}

ce_status_t ce_bce_with_logits_backward(const ce_tensor_t* logits, const ce_tensor_t* targets,
                                         const ce_tensor_t* mask, ce_tensor_t* grad) {
    if (!logits || !targets || !grad) return CE_ERR_INVALID_ARG;
    if (logits->numel != targets->numel || logits->numel != grad->numel) return CE_ERR_INVALID_SHAPE;
    if (mask && mask->numel != logits->numel) return CE_ERR_INVALID_SHAPE;

    int64_t n = logits->numel;
    const float* __restrict__ l_data = logits->data;
    const float* __restrict__ t_data = targets->data;
    const float* __restrict__ m_data = mask ? mask->data : NULL;
    float* __restrict__ g_data = grad->data;

    /* Compute n_valid for normalization */
    double n_valid = 0.0;
    if (mask) {
        for (int64_t i = 0; i < n; i++) {
            if (m_data[i] != 0.0f) n_valid += 1.0;
        }
    } else {
        n_valid = (double)n;
    }

    if (n_valid <= 0.0) return CE_OK;

    float scale = 1.0f / (float)n_valid;

    int64_t i = 0;
#if CE_HAS_AVX2
    __m256 vscale = _mm256_set1_ps(scale);
    __m256 vzero = _mm256_setzero_ps();

    for (; i + 7 < n; i += 8) {
        __m256 vl = _mm256_loadu_ps(&l_data[i]);
        __m256 vt = _mm256_loadu_ps(&t_data[i]);

        /* sigmoid(x) */
        __m256 vs = avx2_sigmoid_ps(vl);

        /* grad = (sigmoid(x) - target) * scale */
        __m256 vg = _mm256_mul_ps(_mm256_sub_ps(vs, vt), vscale);

        /* Apply mask if present */
        if (m_data) {
            __m256 vm = _mm256_loadu_ps(&m_data[i]);
            __m256 mask_cmp = _mm256_cmp_ps(vm, vzero, _CMP_NEQ_OQ);
            vg = _mm256_and_ps(vg, mask_cmp);
        }

        _mm256_storeu_ps(&g_data[i], vg);
    }
#endif
    /* Scalar tail */
    for (; i < n; i++) {
        if (m_data && m_data[i] == 0.0f) {
            g_data[i] = 0.0f;
            continue;
        }
        float sig = 1.0f / (1.0f + expf(-l_data[i]));
        g_data[i] = (sig - t_data[i]) * scale;
    }

    return CE_OK;
}
