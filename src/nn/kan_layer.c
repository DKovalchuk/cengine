/* kan_layer.c — KAN Layer: Polynomial Basis + SiLU, AVX2 + OpenMP */
#include "kan_layer.h"
#include "simd.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

/* ---- Xavier uniform init ---- */
static void xavier_init(ce_tensor_t* t, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    uint32_t seed = 42 + (uint32_t)(uintptr_t)t;
    for (int64_t i = 0; i < t->numel; i++) {
        seed = seed * 1103515245u + 12345u;
        float u = (float)(seed & 0x00FFFFFF) / (float)0x01000000;
        t->data[i] = (u * 2.0f - 1.0f) * limit;
    }
}

/* ================================================================
 * Forward — AVX2 polynomial basis + SiLU + weighted sum
 *
 * For each sample n and feature i:
 *   basis = [1, x, x^2, x^3]
 *   spline_val = sum_k c[j,i,k] * basis[k]
 *   phi = w_b[j,i] * SiLU(x) + w_s[j,i] * spline_val
 *   y[j] = sum_i phi + bias[j]
 * ================================================================ */
ce_status_t ce_kan_layer_forward_batch(ce_module_t* mod,
                                        ce_tensor_t* input,
                                        ce_tensor_t* output) {
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;
    ce_kan_layer_t* st = (ce_kan_layer_t*)mod->internal_state;
    if (!st) return CE_ERR_NOT_INITIALIZED;

    int B = (int)input->shape[0];
    int T = (int)input->shape[1];
    int in_f = st->in_features;
    int out_f = st->out_features;
    int N = B * T;

    const float* w_b = st->w_b->data;
    const float* w_s = st->w_s->data;
    const float* c_data = st->c->data;
    const float* bias = st->bias->data;
    const float* inp = input->data;
    float* out = output->data;

    /* Allocate saved state */
    if (st->saved_basis) ce_tensor_release(st->saved_basis);
    if (st->saved_silu) ce_tensor_release(st->saved_silu);
    if (st->saved_x_norm) ce_tensor_release(st->saved_x_norm);
    if (st->saved_c64) ce_tensor_release(st->saved_c64);

    int64_t basis_s[3] = {N, in_f, 4};
    int64_t silu_s[2] = {N, in_f};
    int64_t xn_s[3] = {B, T, in_f};
    int64_t c64_s[3] = {out_f, in_f, 4};
    st->saved_basis = ce_tensor_zeros(basis_s, 3);
    st->saved_silu = ce_tensor_zeros(silu_s, 2);
    st->saved_x_norm = ce_tensor_create(xn_s, 3, CE_FLOAT32);
    st->saved_c64 = ce_tensor_create(c64_s, 3, CE_FLOAT32);

    if (!st->saved_basis || !st->saved_silu ||
        !st->saved_x_norm || !st->saved_c64)
        return CE_ERR_OUT_OF_MEMORY;

    /* Copy input for saved_x_norm (after normalization) */
    float* xn = st->saved_x_norm->data;
    float* basis = st->saved_basis->data;
    float* silu = st->saved_silu->data;

    /* Normalize input: x_norm = x * 2 - 1 */
    for (int64_t i = 0; i < N * in_f; i++)
        xn[i] = inp[i] * 2.0f - 1.0f;

    /* Save c as float32 (copy from param) */
    memcpy(st->saved_c64->data, c_data, (size_t)out_f * in_f * 4 * sizeof(float));

    /* Compute polynomial basis + SiLU + output */
    #pragma omp parallel for schedule(dynamic) if(N >= 32)
    for (int n = 0; n < N; n++) {
        const float* x_n = xn + n * in_f;
        float* b_n = basis + n * in_f * 4;
        float* s_n = silu + n * in_f;

        int i = 0;
#if CE_HAS_AVX2
        for (; i + 7 < in_f; i += 8) {
            __m256 x = _mm256_loadu_ps(x_n + i);

            /* Polynomial basis: [1, x, x^2, x^3] */
            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 x3 = _mm256_mul_ps(x2, x);

            /* Store basis interleaved: basis[n,i,0..3] */
            __m256 one = _mm256_set1_ps(1.0f);
            /* basis[i*4+0] = 1, basis[i*4+1] = x, basis[i*4+2] = x2, basis[i*4+3] = x3
               But we store as [N, in_f, 4] so for 8 consecutive i values: */
            for (int k = 0; k < 8; k++) {
                b_n[(i + k) * 4 + 0] = 1.0f;
                b_n[(i + k) * 4 + 1] = x_n[i + k];
                b_n[(i + k) * 4 + 2] = x_n[i + k] * x_n[i + k];
                b_n[(i + k) * 4 + 3] = x_n[i + k] * x_n[i + k] * x_n[i + k];
            }

            /* SiLU: x * sigmoid(x) */
            __m256 s = avx2_sigmoid_ps(x);
            __m256 sily = _mm256_mul_ps(x, s);
            _mm256_storeu_ps(s_n + i, sily);
        }
#endif
        for (; i < in_f; i++) {
            float x = x_n[i];
            b_n[i * 4 + 0] = 1.0f;
            b_n[i * 4 + 1] = x;
            b_n[i * 4 + 2] = x * x;
            b_n[i * 4 + 3] = x * x * x;
            float sig = ce_sigmoid_scalar(x);
            s_n[i] = x * sig;
        }
    }

    /* Output: y[n,j] = sum_i (w_b[j,i]*silu[n,i] + w_s[j,i]*spline[n,j,i]) + bias[j]
       where spline[n,j,i] = sum_k c[j,i,k]*basis[n,i,k] */
    #pragma omp parallel for schedule(dynamic) if(N >= 32)
    for (int n = 0; n < N; n++) {
        float* y_n = out + n * out_f;
        const float* b_n = basis + n * in_f * 4;
        const float* s_n = silu + n * in_f;

        /* Initialize with bias */
        for (int j = 0; j < out_f; j++)
            y_n[j] = bias[j];

        /* Accumulate contributions from each input feature */
        for (int i = 0; i < in_f; i++) {
            float si = s_n[i];
            const float* b_ik = b_n + i * 4;

            int j = 0;
#if CE_HAS_AVX2
            for (; j + 3 < out_f; j += 4) {
                /* Load w_b/w_s for j..j+3 in the SAME lane layout as ck/sv
                 * below: lane k and k+4 = element (j+k).  The old
                 * duplicated-pair layout here vs sequential layout in ck
                 * mixed w_b of one output with w_s*spline of another, so
                 * y[j+1..j+3] got wrong phi and j+3 was dropped. */
                __m256 wb = _mm256_set_ps(
                    w_b[(j+3)*in_f+i], w_b[(j+2)*in_f+i],
                    w_b[(j+1)*in_f+i], w_b[(j+0)*in_f+i],
                    w_b[(j+3)*in_f+i], w_b[(j+2)*in_f+i],
                    w_b[(j+1)*in_f+i], w_b[(j+0)*in_f+i]);
                __m256 ws = _mm256_set_ps(
                    w_s[(j+3)*in_f+i], w_s[(j+2)*in_f+i],
                    w_s[(j+1)*in_f+i], w_s[(j+0)*in_f+i],
                    w_s[(j+3)*in_f+i], w_s[(j+2)*in_f+i],
                    w_s[(j+1)*in_f+i], w_s[(j+0)*in_f+i]);

                /* Compute spline_val for j..j+3: sum_k c[j,i,k]*basis[i,k] */
                __m256 sv = _mm256_setzero_ps();
                for (int k = 0; k < 4; k++) {
                    __m256 bk = _mm256_set1_ps(b_ik[k]);
                    __m256 ck = _mm256_set_ps(
                        c_data[(j+3)*in_f*4+i*4+k], c_data[(j+2)*in_f*4+i*4+k],
                        c_data[(j+1)*in_f*4+i*4+k], c_data[(j+0)*in_f*4+i*4+k],
                        c_data[(j+3)*in_f*4+i*4+k], c_data[(j+2)*in_f*4+i*4+k],
                        c_data[(j+1)*in_f*4+i*4+k], c_data[(j+0)*in_f*4+i*4+k]);
                    sv = _mm256_fmadd_ps(ck, bk, sv);
                }

                /* phi = w_b * silu + w_s * spline_val (lanes 0..3 = j..j+3) */
                __m256 phi = _mm256_add_ps(
                    _mm256_mul_ps(wb, _mm256_set1_ps(si)),
                    _mm256_mul_ps(ws, sv));

                /* Accumulate to y[j..j+3] — but y is [out_f], need to scatter */
                float phi_arr[8] __attribute__((aligned(32)));
                _mm256_store_ps(phi_arr, phi);
                for (int jj = 0; jj < 4 && j+jj < out_f; jj++)
                    y_n[j+jj] += phi_arr[jj];
            }
#endif
            for (; j < out_f; j++) {
                float spline_val = 0.0f;
                for (int k = 0; k < 4; k++)
                    spline_val += c_data[j*in_f*4 + i*4 + k] * b_ik[k];
                y_n[j] += w_b[j*in_f + i] * si + w_s[j*in_f + i] * spline_val;
            }
        }
    }

    return CE_OK;
}

ce_status_t ce_kan_layer_forward(ce_module_t* mod,
                                  ce_tensor_t* input,
                                  ce_tensor_t* output) {
    return ce_kan_layer_forward_batch(mod, input, output);
}

/* ================================================================
 * Backward — Analytical polynomial gradient
 *
 * d_basis = [0, 1, 2x, 3x^2]
 * d_spline = c * w_s · d_basis
 * d_base = w_b · d_silu
 * grad_input = grad_output · (d_base + d_spline) * 2
 * ================================================================ */
ce_status_t ce_kan_layer_backward_batch(ce_module_t* mod,
                                         ce_tensor_t* grad_output,
                                         ce_tensor_t* grad_input) {
    if (!mod || !grad_output) return CE_ERR_INVALID_ARG;
    ce_kan_layer_t* st = (ce_kan_layer_t*)mod->internal_state;
    if (!st || !st->saved_basis) return CE_ERR_NOT_INITIALIZED;

    int B = (int)grad_output->shape[0];
    int T = (int)grad_output->shape[1];
    int in_f = st->in_features;
    int out_f = st->out_features;
    int N = B * T;

    const float* g_out = grad_output->data;
    const float* basis = st->saved_basis->data;
    const float* silu = st->saved_silu->data;
    const float* xn = st->saved_x_norm->data;
    const float* w_b = st->w_b->data;
    const float* w_s = st->w_s->data;
    const float* c_data = st->c->data;

    /* Zero gradients */
    ce_tensor_zero(st->g_w_b);
    ce_tensor_zero(st->g_w_s);
    ce_tensor_zero(st->g_c);
    ce_tensor_zero(st->g_bias);
    if (grad_input) ce_tensor_zero(grad_input);

    float* g_w_b = st->g_w_b->data;
    float* g_w_s = st->g_w_s->data;
    float* g_c = st->g_c->data;
    float* g_bias = st->g_bias->data;
    float* g_in = grad_input ? grad_input->data : NULL;

    /* Compute d_silu = sigmoid(x) * (1 + x * (1 - sigmoid(x))) */
    float* d_silu = (float*)malloc((size_t)N * in_f * sizeof(float));
    for (int n = 0; n < N; n++) {
        int i = 0;
#if CE_HAS_AVX2
        for (; i + 7 < in_f; i += 8) {
            __m256 x = _mm256_loadu_ps(xn + n * in_f + i);
            __m256 sig = avx2_sigmoid_ps(x);
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 d = _mm256_mul_ps(sig,
                _mm256_fmadd_ps(x, _mm256_sub_ps(one, sig), one));
            _mm256_storeu_ps(d_silu + n * in_f + i, d);
        }
#endif
        for (; i < in_f; i++) {
            float x = xn[n * in_f + i];
            float sig = ce_sigmoid_scalar(x);
            d_silu[n * in_f + i] = sig * (1.0f + x * (1.0f - sig));
        }
    }

    /* Per-thread gradient buffers */
    int nthreads = omp_get_max_threads();
    float** tg_wb = (float**)malloc((size_t)nthreads * sizeof(float*));
    float** tg_ws = (float**)malloc((size_t)nthreads * sizeof(float*));
    float** tg_c = (float**)malloc((size_t)nthreads * sizeof(float*));
    for (int t = 0; t < nthreads; t++) {
        tg_wb[t] = (float*)calloc((size_t)out_f * in_f, sizeof(float));
        tg_ws[t] = (float*)calloc((size_t)out_f * in_f, sizeof(float));
        tg_c[t] = (float*)calloc((size_t)out_f * in_f * 4, sizeof(float));
    }

    /* Main backward loop */
    #pragma omp parallel for schedule(dynamic) if(N >= 32)
    for (int n = 0; n < N; n++) {
        int tid = omp_get_thread_num();
        float* my_wb = tg_wb[tid];
        float* my_ws = tg_ws[tid];
        float* my_c = tg_c[tid];
        const float* g_n = g_out + n * out_f;
        const float* b_n = basis + n * in_f * 4;
        const float* ds_n = d_silu + n * in_f;
        const float* x_n = xn + n * in_f;

        /* grad_bias += sum_j g[j] */
        for (int j = 0; j < out_f; j++) {
            #pragma omp atomic
            g_bias[j] += g_n[j];
        }

        /* For each input feature i */
        for (int i = 0; i < in_f; i++) {
            float ds_i = ds_n[i];
            const float* b_ik = b_n + i * 4;

            /* d_basis = [0, 1, 2x, 3x^2] */
            float d_b[4] = {0.0f, 1.0f, 2.0f * x_n[i], 3.0f * x_n[i] * x_n[i]};

            for (int j = 0; j < out_f; j++) {
                float g_j = g_n[j];
                int base = j * in_f + i;

                /* g_w_b[j,i] += g[j] * silu[i] */
                my_wb[base] += g_j * silu[n * in_f + i];

                /* spline_val = sum_k c[j,i,k] * basis[i,k] */
                float sv = 0.0f;
                for (int k = 0; k < 4; k++)
                    sv += c_data[base * 4 + k] * b_ik[k];

                /* g_w_s[j,i] += g[j] * spline_val */
                my_ws[base] += g_j * sv;

                /* g_c[j,i,k] += g[j] * w_s[j,i] * d_basis[k] (for grad through basis) */
                /* Actually: g_c is gradient w.r.t. c, which flows through basis in forward */
                /* In backward, g_c accumulates: g[j] * w_s[j,i] * basis[i,k] */
                for (int k = 0; k < 4; k++)
                    my_c[base * 4 + k] += g_j * w_s[base] * b_ik[k];
            }

            /* grad_input[i] += sum_j g[j] * (w_b[j,i]*d_silu[i] + w_s[j,i]*sum_k c[j,i,k]*d_basis[k])
             * — the upstream g[j] must weight every output's contribution
             * (chain rule), as it already does for g_w_b/g_w_s/g_c. */
            if (g_in) {
                float d_spline = 0.0f;
                float d_base = 0.0f;
                for (int j = 0; j < out_f; j++) {
                    float g_j = g_n[j];
                    float dwsc = 0.0f;
                    for (int k = 0; k < 4; k++)
                        dwsc += c_data[(j*in_f+i)*4+k] * d_b[k];
                    d_spline += g_j * w_s[j*in_f+i] * dwsc;
                    d_base += g_j * w_b[j*in_f+i] * ds_i;
                }

                #pragma omp atomic
                g_in[n * in_f + i] += (d_base + d_spline) * 2.0f;
            }
        }
    }

    /* Merge per-thread gradients */
    for (int t = 0; t < nthreads; t++) {
        int64_t ws = (int64_t)out_f * in_f;
        for (int64_t i = 0; i < ws; i++) {
            g_w_b[i] += tg_wb[t][i];
            g_w_s[i] += tg_ws[t][i];
        }
        for (int64_t i = 0; i < ws * 4; i++)
            g_c[i] += tg_c[t][i];
        free(tg_wb[t]); free(tg_ws[t]); free(tg_c[t]);
    }
    free(tg_wb); free(tg_ws); free(tg_c);
    free(d_silu);

    return CE_OK;
}

ce_status_t ce_kan_layer_backward(ce_module_t* mod,
                                   ce_tensor_t* grad_output,
                                   ce_tensor_t* grad_input) {
    return ce_kan_layer_backward_batch(mod, grad_output, grad_input);
}

/* ================================================================
 * Create
 * ================================================================ */
ce_module_t* ce_kan_layer_create(int in_features, int out_features,
                                  const char* name) {
    if (in_features <= 0 || out_features <= 0) return NULL;

    ce_module_t* mod = ce_module_create(name ? name : "kan_layer");
    if (!mod) return NULL;

    ce_kan_layer_t* st = (ce_kan_layer_t*)calloc(1, sizeof(ce_kan_layer_t));
    if (!st) { ce_module_destroy(mod); return NULL; }

    st->in_features = in_features;
    st->out_features = out_features;
    if (name) strncpy(st->name, name, sizeof(st->name) - 1);

    int64_t wb_s[2] = {out_features, in_features};
    int64_t c_s[3] = {out_features, in_features, 4};
    int64_t b_s[1] = {out_features};

    st->w_b = ce_tensor_create(wb_s, 2, CE_FLOAT32);
    st->w_s = ce_tensor_create(wb_s, 2, CE_FLOAT32);
    st->c = ce_tensor_create(c_s, 3, CE_FLOAT32);
    st->bias = ce_tensor_zeros(b_s, 1);

    st->g_w_b = ce_tensor_zeros(wb_s, 2);
    st->g_w_s = ce_tensor_zeros(wb_s, 2);
    st->g_c = ce_tensor_zeros(c_s, 3);
    st->g_bias = ce_tensor_zeros(b_s, 1);

    if (!st->w_b || !st->w_s || !st->c || !st->bias ||
        !st->g_w_b || !st->g_w_s || !st->g_c || !st->g_bias) goto fail;

    /* Init: w_b = 1.0, w_s = small random, c = 0, bias = 0 */
    ce_tensor_fill(st->w_b, 1.0f);
    xavier_init(st->w_s, in_features, out_features);
    /* c stays zero — polynomial part starts neutral */
    /* bias stays zero */

    ce_module_add_parameter(mod, st->w_b, st->g_w_b);
    ce_module_add_parameter(mod, st->w_s, st->g_w_s);
    ce_module_add_parameter(mod, st->c, st->g_c);
    ce_module_add_parameter(mod, st->bias, st->g_bias);

    mod->internal_state = st;
    mod->forward = ce_kan_layer_forward;
    mod->backward = ce_kan_layer_backward;
    return mod;

fail:
    if (st->w_b) ce_tensor_release(st->w_b);
    if (st->w_s) ce_tensor_release(st->w_s);
    if (st->c) ce_tensor_release(st->c);
    if (st->bias) ce_tensor_release(st->bias);
    if (st->g_w_b) ce_tensor_release(st->g_w_b);
    if (st->g_w_s) ce_tensor_release(st->g_w_s);
    if (st->g_c) ce_tensor_release(st->g_c);
    if (st->g_bias) ce_tensor_release(st->g_bias);
    free(st);
    ce_module_destroy(mod);
    return NULL;
}
