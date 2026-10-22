/* kan_recurrent.c — Recurrent KAN ALU: carry loop + 3 KAN layers, AVX2 + OpenMP */
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

/* ================================================================
 * Recurrent KAN forward — full fused carry loop
 *
 * Architecture per timestep t:
 *   input_t = [a_bit, b_bit, carry[t-1], op1, op2] + pos_emb[t]
 *   h1 = KAN1(input_t) + res_proj(input_t) * 0.1
 *   h1 = LayerNorm(h1)
 *   h2 = KAN2(h1) + h1  (residual)
 *   h2 = LayerNorm(h2)
 *   [result, carry_cand] = KAN3(h2)
 *   carry[t] = sigmoid(carry_cand)
 *
 * Parameters:
 *   kan1: KAN layer (5 → H)
 *   kan2: KAN layer (H → H)
 *   kan3: KAN layer (H → 2)
 *   res_proj: [5, H] — input projection for residual
 *   ln1_gamma, ln1_beta: [H] — LayerNorm 1
 *   ln2_gamma, ln2_beta: [H] — LayerNorm 2
 *   pos_emb: [max_T, 5] — positional embedding
 * ================================================================ */

/* Internal state for recurrent KAN */
typedef struct {
    /* KAN layers (3 modules) */
    ce_module_t* kan1;
    ce_module_t* kan2;
    ce_module_t* kan3;

    /* Shared parameters */
    ce_tensor_t* res_proj;      /* [5, H] */
    ce_tensor_t* g_res_proj;
    ce_tensor_t* ln1_gamma;    /* [H] */
    ce_tensor_t* ln1_beta;
    ce_tensor_t* g_ln1_gamma;
    ce_tensor_t* g_ln1_beta;
    ce_tensor_t* ln2_gamma;    /* [H] */
    ce_tensor_t* ln2_beta;
    ce_tensor_t* g_ln2_gamma;
    ce_tensor_t* g_ln2_beta;
    ce_tensor_t* pos_emb;      /* [max_T, 5] */
    ce_tensor_t* g_pos_emb;

    /* Saved state for backward */
    ce_tensor_t* saved_out;        /* [B, T, 2] */
    ce_tensor_t* saved_carry_seq;  /* [B, T] */
    ce_tensor_t* saved_x_t;        /* [B, T, 5] — after carry injection */
    ce_tensor_t* saved_h1_norm;    /* [B, T, H] */
    ce_tensor_t* saved_h2_norm;    /* [B, T, H] */
    ce_tensor_t* saved_m1, *saved_s1;  /* [B, T, 1] */
    ce_tensor_t* saved_m2, *saved_s2;  /* [B, T, 1] */

    int hidden_size;
    int max_len;
    char name[64];
} ce_kan_recurrent_t;

/* ---- Polynomial basis forward (scalar, for single element) ---- */
static inline void poly_basis_4(float x, float out[4]) {
    out[0] = 1.0f;
    out[1] = x;
    out[2] = x * x;
    out[3] = x * x * x;
}

/* ---- Polynomial basis gradient (scalar) ---- */
static inline void poly_basis_grad_4(float x, float out[4]) {
    out[0] = 0.0f;
    out[1] = 1.0f;
    out[2] = 2.0f * x;
    out[3] = 3.0f * x * x;
}

/* ---- KAN forward for single sample [in_f] → [out_f] ---- */
static void kan_forward_1d(
    const float* x,           /* [in_f] normalized input */
    const float* w_b,         /* [out_f, in_f] */
    const float* w_s,         /* [out_f, in_f] */
    const float* c_data,      /* [out_f, in_f, 4] */
    const float* bias,        /* [out_f] */
    float* y,                 /* [out_f] output */
    float* basis_out,         /* [in_f, 4] saved basis */
    float* silu_out,          /* [in_f] saved silu */
    int in_f, int out_f)
{
    /* Compute basis and SiLU */
    for (int i = 0; i < in_f; i++) {
        poly_basis_4(x[i], basis_out + i * 4);
        silu_out[i] = x[i] * ce_sigmoid_scalar(x[i]); /* SiLU: x * sigmoid(x) */
    }

    /* Output: y[j] = sum_i (w_b[j,i]*silu[i] + w_s[j,i]*spline[i]) + bias[j] */
    for (int j = 0; j < out_f; j++) {
        float val = bias[j];
        for (int i = 0; i < in_f; i++) {
            float sv = 0.0f;
            for (int k = 0; k < 4; k++)
                sv += c_data[j*in_f*4 + i*4 + k] * basis_out[i*4 + k];
            val += w_b[j*in_f + i] * silu_out[i] + w_s[j*in_f + i] * sv;
        }
        y[j] = val;
    }
}

/* ---- KAN forward for single sample, batched over out_f with AVX2 ---- */
static void kan_forward_1d_avx2(
    const float* x, const float* w_b, const float* w_s,
    const float* c_data, const float* bias,
    float* y, float* basis_out, float* silu_out,
    int in_f, int out_f)
{
    /* Compute basis and SiLU */
    int i = 0;
#if CE_HAS_AVX2
    for (; i + 7 < in_f; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 x2 = _mm256_mul_ps(xv, xv);
        __m256 x3 = _mm256_mul_ps(x2, xv);
        __m256 sig = avx2_sigmoid_ps(xv);
        __m256 sily = _mm256_mul_ps(xv, sig);
        _mm256_storeu_ps(silu_out + i, sily);
        /* Store basis */
        for (int k = 0; k < 8; k++) {
            basis_out[(i+k)*4+0] = 1.0f;
            basis_out[(i+k)*4+1] = x[i+k];
            basis_out[(i+k)*4+2] = x[i+k]*x[i+k];
            basis_out[(i+k)*4+3] = x[i+k]*x[i+k]*x[i+k];
        }
    }
#endif
    for (; i < in_f; i++) {
        poly_basis_4(x[i], basis_out + i * 4);
        silu_out[i] = x[i] * ce_sigmoid_scalar(x[i]);
    }

    /* Output */
    for (int j = 0; j < out_f; j++) {
        float val = bias[j];
        for (int i = 0; i < in_f; i++) {
            float sv = 0.0f;
            for (int k = 0; k < 4; k++)
                sv += c_data[j*in_f*4 + i*4 + k] * basis_out[i*4 + k];
            val += w_b[j*in_f + i] * silu_out[i] + w_s[j*in_f + i] * sv;
        }
        y[j] = val;
    }
}

/* ---- LayerNorm for single sample [H] → [H] ---- */
static void layer_norm_1d(const float* x, float* y, int H,
                           const float* gamma, const float* beta,
                           float* mean_out, float* std_out) {
    float mean = 0.0f;
    for (int i = 0; i < H; i++) mean += x[i];
    mean /= H;

    float var = 0.0f;
    for (int i = 0; i < H; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float std_val = sqrtf(var / H + 1e-5f);

    for (int i = 0; i < H; i++)
        y[i] = gamma[i] * (x[i] - mean) / std_val + beta[i];

    *mean_out = mean;
    *std_out = std_val;
}

/* ================================================================
 * Forward entry point
 * ================================================================ */
ce_status_t ce_kan_recurrent_forward(ce_module_t* mod,
                                      ce_tensor_t* input,
                                      ce_tensor_t* output) {
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;
    ce_kan_recurrent_t* st = (ce_kan_recurrent_t*)mod->internal_state;
    if (!st) return CE_ERR_NOT_INITIALIZED;

    int B = (int)input->shape[0];
    int T = (int)input->shape[1];
    int H = st->hidden_size;
    int in_f = 5;

    /* Get KAN layer parameters */
    ce_kan_layer_t* k1 = (ce_kan_layer_t*)st->kan1->internal_state;
    ce_kan_layer_t* k2 = (ce_kan_layer_t*)st->kan2->internal_state;
    ce_kan_layer_t* k3 = (ce_kan_layer_t*)st->kan3->internal_state;

    /* Allocate saved state */
    if (st->saved_out) ce_tensor_release(st->saved_out);
    if (st->saved_carry_seq) ce_tensor_release(st->saved_carry_seq);
    if (st->saved_x_t) ce_tensor_release(st->saved_x_t);
    if (st->saved_h1_norm) ce_tensor_release(st->saved_h1_norm);
    if (st->saved_h2_norm) ce_tensor_release(st->saved_h2_norm);
    if (st->saved_m1) ce_tensor_release(st->saved_m1);
    if (st->saved_s1) ce_tensor_release(st->saved_s1);
    if (st->saved_m2) ce_tensor_release(st->saved_m2);
    if (st->saved_s2) ce_tensor_release(st->saved_s2);

    int64_t o_s[3] = {B, T, 2};
    int64_t cs_s[2] = {B, T};
    int64_t xt_s[3] = {B, T, 5};
    int64_t h_s[3] = {B, T, H};
    int64_t ms_s[3] = {B, T, 1};

    st->saved_out = ce_tensor_create(o_s, 3, CE_FLOAT32);
    st->saved_carry_seq = ce_tensor_zeros(cs_s, 2);
    st->saved_x_t = ce_tensor_create(xt_s, 3, CE_FLOAT32);
    st->saved_h1_norm = ce_tensor_create(h_s, 3, CE_FLOAT32);
    st->saved_h2_norm = ce_tensor_create(h_s, 3, CE_FLOAT32);
    st->saved_m1 = ce_tensor_create(ms_s, 3, CE_FLOAT32);
    st->saved_s1 = ce_tensor_create(ms_s, 3, CE_FLOAT32);
    st->saved_m2 = ce_tensor_create(ms_s, 3, CE_FLOAT32);
    st->saved_s2 = ce_tensor_create(ms_s, 3, CE_FLOAT32);

    float* out_data = st->saved_out->data;
    float* carry_data = st->saved_carry_seq->data;
    float* xt_data = st->saved_x_t->data;
    float* h1n_data = st->saved_h1_norm->data;
    float* h2n_data = st->saved_h2_norm->data;
    float* m1_data = st->saved_m1->data;
    float* s1_data = st->saved_s1->data;
    float* m2_data = st->saved_m2->data;
    float* s2_data = st->saved_s2->data;

    /* Per-thread temporary buffers: the batch loop below runs under
     * "#pragma omp parallel for", so shared scratch (x_t, h1_*, h2_*,
     * y3, basis_buf, silu_buf) would be written concurrently by every
     * thread — a data race corrupting outputs nondeterministically. */
    int n_threads = omp_get_max_threads();
    int max_f = (in_f > H) ? in_f : H;

    float* x_t_all    = (float*)malloc((size_t)n_threads * in_f * sizeof(float));
    float* h1_raw_all = (float*)malloc((size_t)n_threads * H * sizeof(float));
    float* h1_res_all = (float*)malloc((size_t)n_threads * H * sizeof(float));
    float* h1_n_all   = (float*)malloc((size_t)n_threads * H * sizeof(float));
    float* h2_raw_all = (float*)malloc((size_t)n_threads * H * sizeof(float));
    float* h2_res_all = (float*)malloc((size_t)n_threads * H * sizeof(float));
    float* h2_n_all   = (float*)malloc((size_t)n_threads * H * sizeof(float));
    float* y3_all     = (float*)malloc((size_t)n_threads * 2 * sizeof(float));
    float* basis_all  = (float*)malloc((size_t)n_threads * max_f * 4 * sizeof(float));
    float* silu_all   = (float*)malloc((size_t)n_threads * max_f * sizeof(float));

    if (!x_t_all || !h1_raw_all || !h1_res_all || !h1_n_all ||
        !h2_raw_all || !h2_res_all || !h2_n_all || !y3_all ||
        !basis_all || !silu_all) {
        free(x_t_all); free(h1_raw_all); free(h1_res_all); free(h1_n_all);
        free(h2_raw_all); free(h2_res_all); free(h2_n_all); free(y3_all);
        free(basis_all); free(silu_all);
        return CE_ERR_OUT_OF_MEMORY;
    }

    /* Main carry loop — sequential across timesteps, parallel across batch */
    for (int t = 0; t < T; t++) {
        #pragma omp parallel for schedule(dynamic) if(B >= 4)
        for (int b = 0; b < B; b++) {
            int tid = omp_get_thread_num();
            float* x_t       = x_t_all    + (size_t)tid * in_f;
            float* h1_raw    = h1_raw_all + (size_t)tid * H;
            float* h1_res    = h1_res_all + (size_t)tid * H;
            float* h1_n      = h1_n_all   + (size_t)tid * H;
            float* h2_raw    = h2_raw_all + (size_t)tid * H;
            float* h2_res    = h2_res_all + (size_t)tid * H;
            float* h2_n      = h2_n_all   + (size_t)tid * H;
            float* y3        = y3_all     + (size_t)tid * 2;
            float* basis_buf = basis_all  + (size_t)tid * max_f * 4;
            float* silu_buf  = silu_all   + (size_t)tid * max_f;

            const float* inp_bt = input->data + ((int64_t)b * T + t) * in_f;
            float* carry_t = carry_data + (int64_t)b * T + t;
            float carry_val = (t == 0) ? 0.0f : carry_data[(int64_t)b * T + t - 1];

            /* Build input: [a_bit, b_bit, carry, op1, op2] + pos_emb */
            for (int i = 0; i < in_f; i++)
                x_t[i] = inp_bt[i] * 2.0f - 1.0f;  /* normalize to [-1,1] */
            x_t[2] = carry_val * 2.0f - 1.0f;  /* carry also normalized */
            /* Add positional embedding */
            if (st->pos_emb) {
                const float* pe = st->pos_emb->data + (int64_t)t * in_f;
                for (int i = 0; i < in_f; i++)
                    x_t[i] += pe[i];
            }

            /* Save x_t */
            float* xt_bt = xt_data + ((int64_t)b * T + t) * in_f;
            for (int i = 0; i < in_f; i++) xt_bt[i] = x_t[i];

            /* KAN1: [in_f] → [H] */
            kan_forward_1d_avx2(x_t, k1->w_b->data, k1->w_s->data,
                                k1->c->data, k1->bias->data,
                                h1_raw, basis_buf, silu_buf, in_f, H);

            /* Residual: h1 = kan1(x) + res_proj(x) * 0.1 */
            const float* rp = st->res_proj->data;
            for (int j = 0; j < H; j++) {
                float proj = 0.0f;
                for (int i = 0; i < in_f; i++)
                    proj += rp[j * in_f + i] * x_t[i];
                h1_res[j] = h1_raw[j] + proj * 0.1f;
            }

            /* LayerNorm1 */
            float mean1, std1;
            layer_norm_1d(h1_res, h1_n, H, st->ln1_gamma->data,
                          st->ln1_beta->data, &mean1, &std1);

            /* Save h1_norm, mean, std */
            float* h1n_bt = h1n_data + ((int64_t)b * T + t) * H;
            for (int j = 0; j < H; j++) h1n_bt[j] = h1_n[j];
            m1_data[(int64_t)b * T + t] = mean1;
            s1_data[(int64_t)b * T + t] = std1;

            /* KAN2: [H] → [H] */
            kan_forward_1d_avx2(h1_n, k2->w_b->data, k2->w_s->data,
                                k2->c->data, k2->bias->data,
                                h2_raw, basis_buf, silu_buf, H, H);

            /* Residual: h2 = kan2(h1) + h1 */
            for (int j = 0; j < H; j++)
                h2_res[j] = h2_raw[j] + h1_n[j];

            /* LayerNorm2 */
            float mean2, std2;
            layer_norm_1d(h2_res, h2_n, H, st->ln2_gamma->data,
                          st->ln2_beta->data, &mean2, &std2);

            /* Save h2_norm, mean, std */
            float* h2n_bt = h2n_data + ((int64_t)b * T + t) * H;
            for (int j = 0; j < H; j++) h2n_bt[j] = h2_n[j];
            m2_data[(int64_t)b * T + t] = mean2;
            s2_data[(int64_t)b * T + t] = std2;

            /* KAN3: [H] → [2] */
            kan_forward_1d_avx2(h2_n, k3->w_b->data, k3->w_s->data,
                                k3->c->data, k3->bias->data,
                                y3, basis_buf, silu_buf, H, 2);

            /* Save output */
            float* out_bt = out_data + ((int64_t)b * T + t) * 2;
            out_bt[0] = y3[0];  /* result logit */
            out_bt[1] = y3[1];  /* carry logit */

            /* Compute carry for next timestep */
            float carry_cand = y3[1];
            if (carry_cand > 20.0f) carry_cand = 20.0f;
            if (carry_cand < -20.0f) carry_cand = -20.0f;
            *carry_t = 1.0f / (1.0f + ce_expf_scalar(-carry_cand));
        }
    }

    /* Copy output to output tensor */
    memcpy(output->data, out_data, (size_t)B * T * 2 * sizeof(float));

    /* Cleanup temps */
    free(x_t_all); free(h1_raw_all); free(h1_res_all); free(h1_n_all);
    free(h2_raw_all); free(h2_res_all); free(h2_n_all); free(y3_all);
    free(basis_all); free(silu_all);

    return CE_OK;
}

/* ================================================================
 * Create
 * ================================================================ */
ce_module_t* ce_kan_recurrent_create(int hidden_size, const char* name) {
    if (hidden_size <= 0) return NULL;

    ce_module_t* mod = ce_module_create(name ? name : "kan_recurrent");
    if (!mod) return NULL;

    ce_kan_recurrent_t* st = (ce_kan_recurrent_t*)calloc(1, sizeof(ce_kan_recurrent_t));
    if (!st) { ce_module_destroy(mod); return NULL; }

    st->hidden_size = hidden_size;
    st->max_len = 256;
    if (name) strncpy(st->name, name, sizeof(st->name) - 1);

    /* Create 3 KAN layers */
    st->kan1 = ce_kan_layer_create(5, hidden_size, "kan1");
    st->kan2 = ce_kan_layer_create(hidden_size, hidden_size, "kan2");
    st->kan3 = ce_kan_layer_create(hidden_size, 2, "kan3");

    if (!st->kan1 || !st->kan2 || !st->kan3) goto fail;

    /* Shared parameters */
    int64_t rp_s[2] = {5, hidden_size};
    int64_t ln_s[1] = {hidden_size};
    int64_t pe_s[2] = {256, 5};

    st->res_proj = ce_tensor_create(rp_s, 2, CE_FLOAT32);
    st->g_res_proj = ce_tensor_zeros(rp_s, 2);
    st->ln1_gamma = ce_tensor_ones(ln_s, 1);
    st->ln1_beta = ce_tensor_zeros(ln_s, 1);
    st->g_ln1_gamma = ce_tensor_zeros(ln_s, 1);
    st->g_ln1_beta = ce_tensor_zeros(ln_s, 1);
    st->ln2_gamma = ce_tensor_ones(ln_s, 1);
    st->ln2_beta = ce_tensor_zeros(ln_s, 1);
    st->g_ln2_gamma = ce_tensor_zeros(ln_s, 1);
    st->g_ln2_beta = ce_tensor_zeros(ln_s, 1);
    st->pos_emb = ce_tensor_zeros(pe_s, 2);
    st->g_pos_emb = ce_tensor_zeros(pe_s, 2);

    if (!st->res_proj || !st->ln1_gamma || !st->ln2_gamma ||
        !st->pos_emb) goto fail;

    /* Register parameters: kan1 + kan2 + kan3 params + shared params */
    /* KAN layers register their own params internally */
    ce_module_add_parameter(mod, st->res_proj, st->g_res_proj);
    ce_module_add_parameter(mod, st->ln1_gamma, st->g_ln1_gamma);
    ce_module_add_parameter(mod, st->ln1_beta, st->g_ln1_beta);
    ce_module_add_parameter(mod, st->ln2_gamma, st->g_ln2_gamma);
    ce_module_add_parameter(mod, st->ln2_beta, st->g_ln2_beta);
    ce_module_add_parameter(mod, st->pos_emb, st->g_pos_emb);

    /* Also register KAN layer params */
    {
        int n1 = ce_module_param_count(st->kan1);
        int n2 = ce_module_param_count(st->kan2);
        int n3 = ce_module_param_count(st->kan3);
        for (int i = 0; i < n1; i++)
            ce_module_add_parameter(mod,
                ce_module_get_parameter(st->kan1, i),
                ce_module_get_gradient(st->kan1, i));
        for (int i = 0; i < n2; i++)
            ce_module_add_parameter(mod,
                ce_module_get_parameter(st->kan2, i),
                ce_module_get_gradient(st->kan2, i));
        for (int i = 0; i < n3; i++)
            ce_module_add_parameter(mod,
                ce_module_get_parameter(st->kan3, i),
                ce_module_get_gradient(st->kan3, i));
    }

    mod->internal_state = st;
    mod->forward = ce_kan_recurrent_forward;
    /* backward not implemented yet — will be done via Python */
    return mod;

fail:
    /* TODO: cleanup */
    free(st);
    ce_module_destroy(mod);
    return NULL;
}
