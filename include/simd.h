#ifndef CENGINE_SIMD_H
#define CENGINE_SIMD_H

#include <immintrin.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compile-time SIMD capability check */
#if defined(__AVX2__) && defined(__FMA__)
  #define CE_HAS_AVX2 1
#else
  #define CE_HAS_AVX2 0
#endif

/* AVX2 element count */
#define CE_AVX2_FLOATS 8
#define CE_AVX2_BYTES  32

/* ---- Scalar math helpers ---- */

static inline float ce_expf_scalar(float x) {
    /* Minimax polynomial approximation for exp(x), |error| < 2^-20
       Range reduction: exp(x) = 2^(x/ln2) = 2^n * exp(r)
       where n = floor(x/ln2), r = x - n*ln2 */
    const float ln2 = 0.6931471805599453f;
    const float inv_ln2 = 1.4426950408889634f;
    /* Clamp to avoid overflow */
    if (x > 88.0f) return 3.402823466e+38f;
    if (x < -87.0f) return 0.0f;

    float z = x * inv_ln2;
    float n_f;
    /* Round to nearest integer */
    n_f = (z >= 0.0f) ? (int)(z + 0.5f) : (int)(z - 0.5f);
    int n = (int)n_f;
    float r = x - n_f * ln2;

    /* Polynomial for exp(r), r in [-ln2/2, ln2/2] */
    float r2 = r * r;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * 0.041666667f)));
    /* More accurate: add r^5 and r^6 terms */
    p += r2 * r2 * (r * 0.008333333f + r2 * 0.001388889f);

    /* Reconstruct: exp(x) = 2^n * p */
    /* Use ldexpf-like bit manipulation for 2^n */
    union { float f; int32_t i; } u;
    u.i = (n + 127) << 23;
    return u.f * p;
}

static inline float ce_sigmoid_scalar(float x) {
    if (x >= 0.0f) {
        float e = ce_expf_scalar(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = ce_expf_scalar(x);
        return e / (1.0f + e);
    }
}

static inline float ce_elu_scalar(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * (ce_expf_scalar(x) - 1.0f);
}

static inline float ce_tanh_scalar(float x) {
    /* Rational approximation from Cephes library */
    if (x > 4.97f) return 1.0f;
    if (x < -4.97f) return -1.0f;
    float x2 = x * x;
    float num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return num / den;
}

/* ---- AVX2 math ---- */

#if CE_HAS_AVX2

static inline __m256 avx2_exp_ps(__m256 x) {
    /* Clamp */
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
    __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);

    __m256 z = _mm256_mul_ps(x, inv_ln2);
    /* Round to nearest */
    __m256 n_f = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2));

    /* Polynomial: 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
    __m256 c6 = _mm256_set1_ps(1.0f / 720.0f);
    __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);
    __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
    __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
    __m256 c2 = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);

    __m256 p = _mm256_fmadd_ps(c6, r, c5);
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, one);
    p = _mm256_fmadd_ps(p, r, one);

    /* 2^n via integer manipulation */
    __m256i n_i = _mm256_cvtps_epi32(n_f);
    n_i = _mm256_add_epi32(n_i, _mm256_set1_epi32(127));
    n_i = _mm256_slli_epi32(n_i, 23);
    __m256 pow2n = _mm256_castsi256_ps(n_i);

    return _mm256_mul_ps(pow2n, p);
}

static inline __m256 avx2_elu_ps(__m256 x, float alpha) {
    __m256 zero = _mm256_setzero_ps();
    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    __m256 exp_x = avx2_exp_ps(x);
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 neg = _mm256_mul_ps(alpha_v, _mm256_sub_ps(exp_x, _mm256_set1_ps(1.0f)));
    return _mm256_blendv_ps(neg, x, mask);
}

static inline __m256 avx2_sigmoid_ps(__m256 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = avx2_exp_ps(neg_x);
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
}

static inline __m256 avx2_tanh_ps(__m256 x) {
    __m256 sigmoid_2x = avx2_sigmoid_ps(_mm256_mul_ps(x, _mm256_set1_ps(2.0f)));
    return _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), sigmoid_2x),
                          _mm256_set1_ps(1.0f));
}

#endif /* CE_HAS_AVX2 */

/* ---- Software prefetch ---- */

#if defined(__GNUC__) || defined(__clang__)
  #define ce_prefetch_read(addr)  __builtin_prefetch((addr), 0, 3)
  #define ce_prefetch_write(addr) __builtin_prefetch((addr), 1, 3)
#else
  #define ce_prefetch_read(addr)  ((void)0)
  #define ce_prefetch_write(addr) ((void)0)
#endif

/* ---- Aligned loads (guarded) ---- */

#if CE_HAS_AVX2
  #define ce_mm256_load(ptr)    _mm256_load_ps(ptr)
  #define ce_mm256_store(ptr,v) _mm256_store_ps(ptr, v)
#else
  #define ce_mm256_load(ptr)    _mm256_loadu_ps(ptr)
  #define ce_mm256_store(ptr,v) _mm256_storeu_ps(ptr, v)
#endif

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_SIMD_H */
