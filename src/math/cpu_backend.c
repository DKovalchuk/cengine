#include "compute.h"
#include "simd.h"
#include "memory.h"
#include <string.h>
#include <math.h>

/* ---- Blocked matrix multiply with AVX2+FMA ----
 *
 * For our model sizes (max ~64x64), matrices fit in L2 cache entirely.
 * We use a simple 2-level blocking: outer tiles for L2, inner 8x8 micro-kernel.
 *
 * C[M,N] = A[M,K] * B[K,N]  (with optional transpose)
 */

#if CE_HAS_AVX2

/* Placeholder for standalone micro-kernel (used in future tiled path) */

#endif /* CE_HAS_AVX2 */

/* Main matmul: C[M,N] = A * B (row-major)
 * CE_TRANSPOSE_A: C = A^T * B, where A is [K,M] stored row-major
 * CE_TRANSPOSE_B: C = A * B^T, where B is [N,K] stored row-major */
static ce_status_t cpu_matmul(const ce_tensor_t* a, const ce_tensor_t* b,
                               ce_tensor_t* out, ce_transpose_t trans) {
    if (!a || !b || !out) return CE_ERR_INVALID_ARG;

    const float* A = a->data;
    const float* B = b->data;
    float* C = out->data;
    int M, N, K, lda, ldb, ldc;

    bool transpose_a = (trans & CE_TRANSPOSE_A) != 0;
    bool transpose_b = (trans & CE_TRANSPOSE_B) != 0;

    /* Determine actual matrix dimensions and access patterns */
    if (transpose_a) {
        /* A is stored as [K,M] (row-major). A^T is [M,K].
           To read A^T[i][k], we read A[k][i] = A[k*lda + i] where lda=M */
        K = (int)a->shape[0];  /* rows of stored A = cols of A^T */
        M = (int)a->shape[1];  /* cols of stored A = rows of A^T */
        lda = (int)a->shape[1]; /* leading dim of stored A */
    } else {
        M = (int)a->shape[0];
        K = (int)a->shape[1];
        lda = (int)a->shape[1];
    }

    if (transpose_b) {
        /* B is stored as [N,K] (row-major). B^T is [K,N].
           To read B^T[k][j], we read B[j][k] = B[j*ldb + k] where ldb=K */
        N = (int)b->shape[0];
        ldb = (int)b->shape[1]; /* row stride of stored [N,K] is K */
    } else {
        N = (int)b->shape[1];
        ldb = (int)b->shape[1];
    }

    ldc = N;

    /* Inner dimensions must agree: the reduction over K indexes the larger
     * of A/B along K — a mismatch reads past the smaller operand. */
    int64_t b_inner = transpose_b ? b->shape[1] : b->shape[0];
    if (a->ndim != 2 || b->ndim != 2 ||
        b_inner != K ||
        out->shape[0] != M || out->shape[1] != N) {
        return CE_ERR_INVALID_SHAPE;
    }

#if CE_HAS_AVX2
    /* Zero output */
    memset(C, 0, (size_t)M * (size_t)N * sizeof(float));

    /* AVX2 blocked matmul with proper transpose handling */
    for (int i = 0; i + 7 < M; i += 8) {
        for (int j = 0; j + 7 < N; j += 8) {
            __m256 c0 = _mm256_loadu_ps(&C[i * ldc + j]);
            __m256 c1 = _mm256_loadu_ps(&C[(i+1) * ldc + j]);
            __m256 c2 = _mm256_loadu_ps(&C[(i+2) * ldc + j]);
            __m256 c3 = _mm256_loadu_ps(&C[(i+3) * ldc + j]);
            __m256 c4 = _mm256_loadu_ps(&C[(i+4) * ldc + j]);
            __m256 c5 = _mm256_loadu_ps(&C[(i+5) * ldc + j]);
            __m256 c6 = _mm256_loadu_ps(&C[(i+6) * ldc + j]);
            __m256 c7 = _mm256_loadu_ps(&C[(i+7) * ldc + j]);

            for (int k = 0; k < K; k++) {
                __m256 b_vec;

                if (transpose_b) {
                    /* B^T[k,j..j+7] = B[j..j+7, k] — gather column k of stored B */
                    /* This is not contiguous, use scalar broadcast for B elements */
                    for (int jj = 0; jj < 8; jj++) {
                        ((float*)&b_vec)[jj] = B[(j+jj) * ldb + k];
                    }
                } else {
                    b_vec = _mm256_loadu_ps(&B[k * ldb + j]);
                }

                __m256 a;
                if (transpose_a) {
                    /* A^T[i,k] = A[k,i] — read from stored A[k, i..i+7] */
                    a = _mm256_loadu_ps(&A[k * lda + i]);
                } else {
                    a = _mm256_broadcast_ss(&A[i * lda + k]);
                }

                if (transpose_a) {
                    /* A^T rows are contiguous — need to multiply each A^T[i+ii][k] by b_vec */
                    /* Unroll: each a[ii] is a different element */
                    __m256 a0 = _mm256_set1_ps(((float*)&a)[0]);
                    __m256 a1 = _mm256_set1_ps(((float*)&a)[1]);
                    __m256 a2 = _mm256_set1_ps(((float*)&a)[2]);
                    __m256 a3 = _mm256_set1_ps(((float*)&a)[3]);
                    __m256 a4 = _mm256_set1_ps(((float*)&a)[4]);
                    __m256 a5 = _mm256_set1_ps(((float*)&a)[5]);
                    __m256 a6 = _mm256_set1_ps(((float*)&a)[6]);
                    __m256 a7 = _mm256_set1_ps(((float*)&a)[7]);
                    c0 = _mm256_fmadd_ps(a0, b_vec, c0);
                    c1 = _mm256_fmadd_ps(a1, b_vec, c1);
                    c2 = _mm256_fmadd_ps(a2, b_vec, c2);
                    c3 = _mm256_fmadd_ps(a3, b_vec, c3);
                    c4 = _mm256_fmadd_ps(a4, b_vec, c4);
                    c5 = _mm256_fmadd_ps(a5, b_vec, c5);
                    c6 = _mm256_fmadd_ps(a6, b_vec, c6);
                    c7 = _mm256_fmadd_ps(a7, b_vec, c7);
                } else {
                    /* Normal: broadcast A[i+ii][k] */
                    a = _mm256_broadcast_ss(&A[i * lda + k]);
                    c0 = _mm256_fmadd_ps(a, b_vec, c0);
                    a = _mm256_broadcast_ss(&A[(i+1) * lda + k]);
                    c1 = _mm256_fmadd_ps(a, b_vec, c1);
                    a = _mm256_broadcast_ss(&A[(i+2) * lda + k]);
                    c2 = _mm256_fmadd_ps(a, b_vec, c2);
                    a = _mm256_broadcast_ss(&A[(i+3) * lda + k]);
                    c3 = _mm256_fmadd_ps(a, b_vec, c3);
                    a = _mm256_broadcast_ss(&A[(i+4) * lda + k]);
                    c4 = _mm256_fmadd_ps(a, b_vec, c4);
                    a = _mm256_broadcast_ss(&A[(i+5) * lda + k]);
                    c5 = _mm256_fmadd_ps(a, b_vec, c5);
                    a = _mm256_broadcast_ss(&A[(i+6) * lda + k]);
                    c6 = _mm256_fmadd_ps(a, b_vec, c6);
                    a = _mm256_broadcast_ss(&A[(i+7) * lda + k]);
                    c7 = _mm256_fmadd_ps(a, b_vec, c7);
                }
            }

            _mm256_storeu_ps(&C[i * ldc + j], c0);
            _mm256_storeu_ps(&C[(i+1) * ldc + j], c1);
            _mm256_storeu_ps(&C[(i+2) * ldc + j], c2);
            _mm256_storeu_ps(&C[(i+3) * ldc + j], c3);
            _mm256_storeu_ps(&C[(i+4) * ldc + j], c4);
            _mm256_storeu_ps(&C[(i+5) * ldc + j], c5);
            _mm256_storeu_ps(&C[(i+6) * ldc + j], c6);
            _mm256_storeu_ps(&C[(i+7) * ldc + j], c7);
        }
    }

    /* Scalar tail for remaining rows */
    for (int i = (M / 8) * 8; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = transpose_a ? A[k * lda + i] : A[i * lda + k];
                float b_val = transpose_b ? B[j * ldb + k] : B[k * ldb + j];
                sum += a_val * b_val;
            }
            C[i * ldc + j] = sum;
        }
    }

    /* Scalar tail for remaining columns */
    for (int i = 0; i < (M / 8) * 8; i++) {
        for (int j = (N / 8) * 8; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = transpose_a ? A[k * lda + i] : A[i * lda + k];
                float b_val = transpose_b ? B[j * ldb + k] : B[k * ldb + j];
                sum += a_val * b_val;
            }
            C[i * ldc + j] = sum;
        }
    }

    return CE_OK;
#else
    /* Scalar fallback with proper transpose */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = transpose_a ? A[k * lda + i] : A[i * lda + k];
                float b_val = transpose_b ? B[j * ldb + k] : B[k * ldb + j];
                sum += a_val * b_val;
            }
            C[i * ldc + j] = sum;
        }
    }
    return CE_OK;
#endif
}

/* ---- Elementwise operations ---- */

static ce_status_t cpu_add(const ce_tensor_t* a, const ce_tensor_t* b, ce_tensor_t* out) {
    if (!a || !b || !out) return CE_ERR_INVALID_ARG;
    if (a->numel != b->numel || a->numel != out->numel) return CE_ERR_INVALID_SHAPE;

    int64_t n = a->numel;
    int64_t i = 0;

#if CE_HAS_AVX2
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        _mm256_storeu_ps(&out->data[i], _mm256_add_ps(va, vb));
    }
#endif
    for (; i < n; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return CE_OK;
}

static ce_status_t cpu_mul(const ce_tensor_t* a, const ce_tensor_t* b, ce_tensor_t* out) {
    if (!a || !b || !out) return CE_ERR_INVALID_ARG;
    if (a->numel != b->numel || a->numel != out->numel) return CE_ERR_INVALID_SHAPE;

    int64_t n = a->numel;
    int64_t i = 0;

#if CE_HAS_AVX2
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        _mm256_storeu_ps(&out->data[i], _mm256_mul_ps(va, vb));
    }
#endif
    for (; i < n; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    return CE_OK;
}

static ce_status_t cpu_axpy(float alpha, const ce_tensor_t* x, ce_tensor_t* y) {
    if (!x || !y) return CE_ERR_INVALID_ARG;
    if (x->numel != y->numel) return CE_ERR_INVALID_SHAPE;

    int64_t n = x->numel;
    int64_t i = 0;

#if CE_HAS_AVX2
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x->data[i]);
        __m256 vy = _mm256_loadu_ps(&y->data[i]);
        _mm256_storeu_ps(&y->data[i], _mm256_fmadd_ps(va, vx, vy));
    }
#endif
    for (; i < n; i++) {
        y->data[i] += alpha * x->data[i];
    }
    return CE_OK;
}

/* ---- Activation forward ---- */

static ce_status_t cpu_activation(const ce_tensor_t* in, ce_activation_t type,
                                   float alpha, float temperature, ce_tensor_t* out) {
    if (!in || !out) return CE_ERR_INVALID_ARG;
    if (in->numel != out->numel) return CE_ERR_INVALID_SHAPE;

    int64_t n = in->numel;
    float* __restrict__ od = out->data;
    const float* __restrict__ id = in->data;

    switch (type) {
    case CE_ACT_ELU: {
        int64_t i = 0;
#if CE_HAS_AVX2
        for (; i + 7 < n; i += 8) {
            __m256 x = _mm256_loadu_ps(&id[i]);
            _mm256_storeu_ps(&od[i], avx2_elu_ps(x, alpha));
        }
#endif
        for (; i < n; i++) od[i] = ce_elu_scalar(id[i], alpha);
        break;
    }
    case CE_ACT_SIGMOID: {
        int64_t i = 0;
#if CE_HAS_AVX2
        for (; i + 7 < n; i += 8) {
            __m256 x = _mm256_loadu_ps(&id[i]);
            _mm256_storeu_ps(&od[i], avx2_sigmoid_ps(x));
        }
#endif
        for (; i < n; i++) od[i] = ce_sigmoid_scalar(id[i]);
        break;
    }
    case CE_ACT_TANH: {
        int64_t i = 0;
#if CE_HAS_AVX2
        for (; i + 7 < n; i += 8) {
            __m256 x = _mm256_loadu_ps(&id[i]);
            _mm256_storeu_ps(&od[i], avx2_tanh_ps(x));
        }
#endif
        for (; i < n; i++) od[i] = ce_tanh_scalar(id[i]);
        break;
    }
    case CE_ACT_RELU: {
        int64_t i = 0;
#if CE_HAS_AVX2
        __m256 zero = _mm256_setzero_ps();
        for (; i + 7 < n; i += 8) {
            __m256 x = _mm256_loadu_ps(&id[i]);
            _mm256_storeu_ps(&od[i], _mm256_max_ps(x, zero));
        }
#endif
        for (; i < n; i++) od[i] = (id[i] > 0.0f) ? id[i] : 0.0f;
        break;
    }
    case CE_ACT_SOFTMAX: {
        /* Softmax over the last dimension. Assumes 2D tensor [batch, classes]. */
        (void)temperature;
        int64_t batch = (in->ndim >= 2) ? in->shape[0] : 1;
        int64_t classes = in->numel / batch;
        for (int64_t b = 0; b < batch; b++) {
            const float* row = &id[b * classes];
            float* orow = &od[b * classes];
            /* Find max for numerical stability */
            float max_val = row[0];
            for (int64_t c = 1; c < classes; c++) {
                if (row[c] > max_val) max_val = row[c];
            }
            /* exp and sum */
            float sum = 0.0f;
            for (int64_t c = 0; c < classes; c++) {
                orow[c] = ce_expf_scalar(row[c] - max_val);
                sum += orow[c];
            }
            /* normalize */
            float inv_sum = 1.0f / sum;
            for (int64_t c = 0; c < classes; c++) {
                orow[c] *= inv_sum;
            }
        }
        break;
    }
    case CE_ACT_GUMBEL_SOFTMAX: {
        /* Gumbel-Softmax STE: compute soft probabilities,
           let the layer code handle hard argmax assignment */
        (void)temperature;
        int64_t batch = (in->ndim >= 2) ? in->shape[0] : 1;
        int64_t classes = in->numel / batch;
        for (int64_t b = 0; b < batch; b++) {
            const float* row = &id[b * classes];
            float* orow = &od[b * classes];
            float max_val = row[0];
            for (int64_t c = 1; c < classes; c++) {
                if (row[c] > max_val) max_val = row[c];
            }
            float sum = 0.0f;
            for (int64_t c = 0; c < classes; c++) {
                orow[c] = ce_expf_scalar((row[c] - max_val) / temperature);
                sum += orow[c];
            }
            float inv_sum = 1.0f / sum;
            for (int64_t c = 0; c < classes; c++) {
                orow[c] *= inv_sum;
            }
        }
        break;
    }
    default:
        return CE_ERR_INVALID_ARG;
    }
    return CE_OK;
}

/* ---- Activation backward ---- */

static ce_status_t cpu_activation_backward(const ce_tensor_t* grad_out,
                                            const ce_tensor_t* act_out,
                                            ce_activation_t type, float alpha,
                                            ce_tensor_t* grad_in) {
    if (!grad_out || !act_out || !grad_in) return CE_ERR_INVALID_ARG;
    int64_t n = grad_out->numel;
    const float* __restrict__ go = grad_out->data;
    const float* __restrict__ ao = act_out->data;
    float* __restrict__ gi = grad_in->data;

    switch (type) {
    case CE_ACT_ELU: {
        /* d/dx ELU(x) = 1 if x > 0, else ELU(x) + alpha */
        int64_t i = 0;
#if CE_HAS_AVX2
        __m256 va = _mm256_set1_ps(alpha);
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 7 < n; i += 8) {
            __m256 g = _mm256_loadu_ps(&go[i]);
            __m256 y = _mm256_loadu_ps(&ao[i]);
            __m256 pos_mask = _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_GT_OQ);
            __m256 deriv = _mm256_blendv_ps(
                _mm256_add_ps(y, va), /* ELU(x) + alpha for x <= 0 */
                one,                   /* 1 for x > 0 */
                pos_mask);
            _mm256_storeu_ps(&gi[i], _mm256_mul_ps(g, deriv));
        }
#endif
        for (; i < n; i++) {
            gi[i] = go[i] * ((ao[i] > 0.0f) ? 1.0f : ao[i] + alpha);
        }
        break;
    }
    case CE_ACT_SIGMOID: {
        /* d/dx sigmoid = sigmoid(x) * (1 - sigmoid(x)) = y * (1-y) */
        int64_t i = 0;
#if CE_HAS_AVX2
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 7 < n; i += 8) {
            __m256 g = _mm256_loadu_ps(&go[i]);
            __m256 y = _mm256_loadu_ps(&ao[i]);
            _mm256_storeu_ps(&gi[i], _mm256_mul_ps(g, _mm256_mul_ps(y, _mm256_sub_ps(one, y))));
        }
#endif
        for (; i < n; i++) {
            gi[i] = go[i] * ao[i] * (1.0f - ao[i]);
        }
        break;
    }
    case CE_ACT_TANH: {
        /* d/dx tanh = 1 - tanh^2(x) = 1 - y^2 */
        int64_t i = 0;
#if CE_HAS_AVX2
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 7 < n; i += 8) {
            __m256 g = _mm256_loadu_ps(&go[i]);
            __m256 y = _mm256_loadu_ps(&ao[i]);
            _mm256_storeu_ps(&gi[i], _mm256_mul_ps(g, _mm256_sub_ps(one, _mm256_mul_ps(y, y))));
        }
#endif
        for (; i < n; i++) {
            gi[i] = go[i] * (1.0f - ao[i] * ao[i]);
        }
        break;
    }
    case CE_ACT_RELU: {
        /* d/dx ReLU = 1 if x > 0, else 0 */
        int64_t i = 0;
#if CE_HAS_AVX2
        __m256 zero = _mm256_setzero_ps();
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 7 < n; i += 8) {
            __m256 g = _mm256_loadu_ps(&go[i]);
            __m256 y = _mm256_loadu_ps(&ao[i]);
            __m256 mask = _mm256_cmp_ps(y, zero, _CMP_GT_OQ);
            _mm256_storeu_ps(&gi[i], _mm256_mul_ps(g, _mm256_and_ps(mask, one)));
        }
#endif
        for (; i < n; i++) {
            gi[i] = go[i] * ((ao[i] > 0.0f) ? 1.0f : 0.0f);
        }
        break;
    }
    case CE_ACT_SOFTMAX: {
        /* Jacobian-vector product for softmax: dy_i * (g_i - sum(g_j * y_j)) */
        int64_t batch = (grad_out->ndim >= 2) ? grad_out->shape[0] : 1;
        int64_t classes = grad_out->numel / batch;
        for (int64_t b = 0; b < batch; b++) {
            const float* grow = &go[b * classes];
            const float* yrow = &ao[b * classes];
            float* girow = &gi[b * classes];
            /* dot = sum(g_j * y_j) */
            float dot = 0.0f;
            for (int64_t c = 0; c < classes; c++) dot += grow[c] * yrow[c];
            for (int64_t c = 0; c < classes; c++) {
                girow[c] = yrow[c] * (grow[c] - dot);
            }
        }
        break;
    }
    default:
        return CE_ERR_INVALID_ARG;
    }
    return CE_OK;
}

/* ---- Reductions ---- */

static ce_status_t cpu_sum(const ce_tensor_t* in, ce_tensor_t* out, int dim) {
    if (!in || !out) return CE_ERR_INVALID_ARG;
    if (dim < 0 || dim >= in->ndim) return CE_ERR_INVALID_ARG;

    int64_t rows = (in->ndim >= 2) ? in->shape[0] : 1;
    int64_t cols = in->numel / rows;

    if (dim == 1) {
        /* Sum over columns → output [rows] */
        for (int64_t r = 0; r < rows; r++) {
            float s = 0.0f;
            for (int64_t c = 0; c < cols; c++) s += in->data[r * cols + c];
            out->data[r] = s;
        }
    } else {
        /* Sum over rows → output [cols] */
        memset(out->data, 0, (size_t)cols * sizeof(float));
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < cols; c++) {
                out->data[c] += in->data[r * cols + c];
            }
        }
    }
    return CE_OK;
}

static ce_status_t cpu_mean(const ce_tensor_t* in, ce_tensor_t* out, int dim) {
    ce_status_t st = cpu_sum(in, out, dim);
    if (st != CE_OK) return st;
    int64_t reduce_size = in->shape[dim];
    float scale = 1.0f / (float)reduce_size;
    int64_t n = out->numel;
    for (int64_t i = 0; i < n; i++) out->data[i] *= scale;
    return CE_OK;
}

static float cpu_norm(const ce_tensor_t* in) {
    if (!in || !in->data) return 0.0f;
    double sum = 0.0;
    for (int64_t i = 0; i < in->numel; i++) {
        double v = (double)in->data[i];
        sum += v * v;
    }
    return (float)sqrt(sum);
}

static float cpu_dot(const ce_tensor_t* a, const ce_tensor_t* b) {
    if (!a || !b || a->numel != b->numel) return 0.0f;
    double sum = 0.0;
    int64_t i = 0;

#if CE_HAS_AVX2
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < a->numel; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    /* Horizontal sum of vsum */
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float partial;
    _mm_store_ss(&partial, s);
    sum = (double)partial;
#endif
    for (; i < a->numel; i++) {
        sum += (double)a->data[i] * (double)b->data[i];
    }
    return (float)sum;
}

static ce_status_t cpu_max_reduce(const ce_tensor_t* in, ce_tensor_t* out,
                                   int dim, ce_tensor_t* indices) {
    if (!in || !out) return CE_ERR_INVALID_ARG;
    int64_t rows = (in->ndim >= 2) ? in->shape[0] : 1;
    int64_t cols = in->numel / rows;

    if (dim == 1) {
        for (int64_t r = 0; r < rows; r++) {
            float best = in->data[r * cols];
            int64_t best_idx = 0;
            for (int64_t c = 1; c < cols; c++) {
                if (in->data[r * cols + c] > best) {
                    best = in->data[r * cols + c];
                    best_idx = c;
                }
            }
            out->data[r] = best;
            if (indices) indices->data[r] = (float)best_idx;
        }
    }
    return CE_OK;
}

static int cpu_get_max_threads(void) {
    return 1; /* single-threaded for now */
}

/* ---- Backend singleton ---- */

static ce_compute_backend_t g_cpu_backend = {
    .name = "cpu",
    .device = CE_DEVICE_CPU,
    .matmul = cpu_matmul,
    .add = cpu_add,
    .mul = cpu_mul,
    .axpy = cpu_axpy,
    .activation = cpu_activation,
    .activation_backward = cpu_activation_backward,
    .sum = cpu_sum,
    .mean = cpu_mean,
    .max = cpu_max_reduce,
    .norm = cpu_norm,
    .dot = cpu_dot,
    .get_max_threads = cpu_get_max_threads,
};

ce_compute_backend_t* ce_cpu_backend_create(void) {
    return &g_cpu_backend;
}
