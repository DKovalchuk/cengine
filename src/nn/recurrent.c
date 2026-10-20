/* recurrent.c — Recurrent MoE ALU Expert with BPTT + carry propagation + MoE routing */
#include "recurrent.h"
#include "simd.h"
#include "memory.h"
#include "autograd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

/* ── Debug helpers (gated by ce_get_config()->enable_debug_logging) ─── */

static float dbg_mean(const float* v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    return (float)(s / n);
}

static float dbg_std(const float* v, int n, float mean_val) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = v[i] - mean_val;
        s += d * d;
    }
    return (float)sqrt(s / n);
}

static float dbg_min(const float* v, int n) {
    float m = v[0];
    for (int i = 1; i < n; i++) if (v[i] < m) m = v[i];
    return m;
}

static float dbg_max(const float* v, int n) {
    float m = v[0];
    for (int i = 1; i < n; i++) if (v[i] > m) m = v[i];
    return m;
}

static float dbg_norm(const float* v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    return (float)sqrt(s);
}

static void xavier_init(ce_tensor_t* t, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    uint32_t seed = 42 + (uint32_t)(uintptr_t)t;
    for (int64_t i = 0; i < t->numel; i++) {
        seed = seed * 1103515245u + 12345u;
        float u = (float)(seed & 0x00FFFFFF) / (float)0x01000000;
        t->data[i] = (u * 2.0f - 1.0f) * limit;
    }
}

ce_module_t* ce_recurrent_expert_create(int hidden_size, const char* name) {
    if (hidden_size <= 0) return NULL;
    ce_module_t* mod = ce_module_create(name ? name : "recurrent_expert");
    if (!mod) return NULL;

    ce_recurrent_expert_t* st = (ce_recurrent_expert_t*)calloc(1, sizeof(ce_recurrent_expert_t));
    if (!st) { ce_module_destroy(mod); return NULL; }

    int H = hidden_size;
    int in = 6 + H;
    st->hidden_size = H;
    st->input_dim = 6;
    if (name) strncpy(st->name, name, sizeof(st->name) - 1);

    int64_t w1_s[2] = {H, in}, b1_s[1] = {H};
    int64_t w2_s[2] = {H, H}, b2_s[1] = {H};
    int64_t wo_s[2] = {2, H}, bo_s[1] = {2};
    int64_t wh_s[2] = {H, H}, bh_s[1] = {H};

    st->w1 = ce_tensor_create(w1_s, 2, CE_FLOAT32); xavier_init(st->w1, in, H);
    st->b1 = ce_tensor_zeros(b1_s, 1);
    st->w2 = ce_tensor_create(w2_s, 2, CE_FLOAT32); xavier_init(st->w2, H, H);
    st->b2 = ce_tensor_zeros(b2_s, 1);
    st->w_out = ce_tensor_create(wo_s, 2, CE_FLOAT32); xavier_init(st->w_out, H, 2);
    st->b_out = ce_tensor_zeros(bo_s, 1);
    st->w_h = ce_tensor_create(wh_s, 2, CE_FLOAT32); xavier_init(st->w_h, H, H);
    st->b_h = ce_tensor_zeros(bh_s, 1);

    st->gw1 = ce_tensor_zeros(w1_s, 2); st->gb1 = ce_tensor_zeros(b1_s, 1);
    st->gw2 = ce_tensor_zeros(w2_s, 2); st->gb2 = ce_tensor_zeros(b2_s, 1);
    st->gw_out = ce_tensor_zeros(wo_s, 2); st->gb_out = ce_tensor_zeros(bo_s, 1);
    st->gw_h = ce_tensor_zeros(wh_s, 2); st->gb_h = ce_tensor_zeros(bh_s, 1);

    ce_module_add_parameter(mod, st->w1, st->gw1);
    ce_module_add_parameter(mod, st->b1, st->gb1);
    ce_module_add_parameter(mod, st->w2, st->gw2);
    ce_module_add_parameter(mod, st->b2, st->gb2);
    ce_module_add_parameter(mod, st->w_out, st->gw_out);
    ce_module_add_parameter(mod, st->b_out, st->gb_out);
    ce_module_add_parameter(mod, st->w_h, st->gw_h);
    ce_module_add_parameter(mod, st->b_h, st->gb_h);

    mod->internal_state = st;
    mod->forward  = ce_recurrent_expert_forward;
    mod->backward = ce_recurrent_expert_backward;
    return mod;
}

/* ---- Forward (inference mode: no saved state) ---- */
ce_status_t ce_recurrent_expert_forward_infer(ce_module_t* mod,
                                               ce_tensor_t* input,
                                               ce_tensor_t* output) {
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;
    ce_recurrent_expert_t* st = (ce_recurrent_expert_t*)mod->internal_state;
    if (!st) return CE_ERR_NOT_INITIALIZED;

    int B = (int)input->shape[0];
    int T = (int)input->shape[1];
    int H = st->hidden_size;
    int in_dim = 6 + H;

    /* Allocate local hidden/carry (not saved to struct) */
    float* hidden = (float*)calloc((size_t)B * H, sizeof(float));
    float* carry = (float*)calloc((size_t)B, sizeof(float));
    float* x5h = (float*)malloc((size_t)in_dim * sizeof(float));
    float* h_new = (float*)malloc((size_t)H * sizeof(float));
    float* logits = (float*)malloc(2 * sizeof(float));
    float* h_cache = (float*)malloc((size_t)H * sizeof(float));
    float* elu1_c = (float*)malloc((size_t)H * sizeof(float));
    float* elu2_c = (float*)malloc((size_t)H * sizeof(float));

    for (int t = 0; t < T; t++) {
        for (int b = 0; b < B; b++) {
            int64_t base = (int64_t)b * T + t;
            const float* inp_t = input->data + base * 6;
            float carry_in = (t == 0) ? 0.0f : carry[b];

            /* Build input */
            float* si = x5h;
            si[0] = inp_t[0]; si[1] = inp_t[1];
            si[2] = carry_in;
            si[3] = inp_t[3]; si[4] = inp_t[4]; si[5] = inp_t[5];
            memcpy(si + 6, hidden + (int64_t)b * H, (size_t)H * sizeof(float));

            /* MLP */
            for (int j = 0; j < H; j++) {
                float sum = st->b1->data[j];
                for (int k = 0; k < in_dim; k++)
                    sum += st->w1->data[j * in_dim + k] * si[k];
                elu1_c[j] = (sum > 0.0f) ? sum : expf(sum) - 1.0f;
            }
            for (int j = 0; j < H; j++) {
                float sum = st->b2->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w2->data[j * H + k] * elu1_c[k];
                elu2_c[j] = (sum > 0.0f) ? sum : expf(sum) - 1.0f;
            }
            for (int j = 0; j < 2; j++) {
                float sum = st->b_out->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w_out->data[j * H + k] * elu2_c[k];
                logits[j] = sum;
            }
            for (int j = 0; j < H; j++) {
                float sum = st->b_h->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w_h->data[j * H + k] * elu2_c[k];
                h_new[j] = tanhf(sum);
            }

            carry[b] = 1.0f / (1.0f + expf(-logits[1]));
            memcpy(hidden + (int64_t)b * H, h_new, (size_t)H * sizeof(float));

            float* out_t = output->data + base * 2;
            out_t[0] = logits[0]; out_t[1] = logits[1];
        }
    }

    free(hidden); free(carry); free(x5h); free(h_new);
    free(logits); free(h_cache); free(elu1_c); free(elu2_c);
    return CE_OK;
}

/* ---- Forward (inference with pre-allocated buffers: zero malloc) ---- */
ce_status_t ce_recurrent_expert_forward_infer_prealloc(
    ce_module_t* mod, ce_tensor_t* input, ce_tensor_t* output,
    float* hidden_buf,   /* [B * H] — pre-zeroed */
    float* carry_buf,    /* [B] — pre-zeroed */
    float* x5h_buf,      /* [6 + H] */
    float* h_new_buf,    /* [H] */
    float* elu1_buf,     /* [H] */
    float* elu2_buf)     /* [H] */
{
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;
    ce_recurrent_expert_t* st = (ce_recurrent_expert_t*)mod->internal_state;
    if (!st) return CE_ERR_NOT_INITIALIZED;

    int B = (int)input->shape[0];
    int T = (int)input->shape[1];
    int H = st->hidden_size;
    int in_dim = 6 + H;

    /* Zero state buffers */
    memset(hidden_buf, 0, (size_t)B * H * sizeof(float));
    memset(carry_buf, 0, (size_t)B * sizeof(float));

    for (int t = 0; t < T; t++) {
        for (int b = 0; b < B; b++) {
            int64_t base = (int64_t)b * T + t;
            const float* inp_t = input->data + base * 6;
            float carry_in = (t == 0) ? 0.0f : carry_buf[b];

            float* si = x5h_buf;
            si[0] = inp_t[0]; si[1] = inp_t[1];
            si[2] = carry_in;
            si[3] = inp_t[3]; si[4] = inp_t[4]; si[5] = inp_t[5];
            memcpy(si + 6, hidden_buf + (int64_t)b * H, (size_t)H * sizeof(float));

            /* Layer 1: elu1 = ELU(W1 * x + b1) */
            for (int j = 0; j < H; j++) {
                float sum = st->b1->data[j];
                for (int k = 0; k < in_dim; k++)
                    sum += st->w1->data[j * in_dim + k] * si[k];
                elu1_buf[j] = (sum > 0.0f) ? sum : expf(sum) - 1.0f;
            }
            /* Layer 2: elu2 = ELU(W2 * elu1 + b2) */
            for (int j = 0; j < H; j++) {
                float sum = st->b2->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w2->data[j * H + k] * elu1_buf[k];
                elu2_buf[j] = (sum > 0.0f) ? sum : expf(sum) - 1.0f;
            }
            /* Output: logits = W_out * elu2 + b_out */
            float logits0, logits1;
            {
                float sum0 = st->b_out->data[0];
                float sum1 = st->b_out->data[1];
                for (int k = 0; k < H; k++) {
                    sum0 += st->w_out->data[0 * H + k] * elu2_buf[k];
                    sum1 += st->w_out->data[1 * H + k] * elu2_buf[k];
                }
                logits0 = sum0;
                logits1 = sum1;
            }
            /* Hidden: new_h = tanh(W_h * elu2 + b_h) */
            for (int j = 0; j < H; j++) {
                float sum = st->b_h->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w_h->data[j * H + k] * elu2_buf[k];
                h_new_buf[j] = tanhf(sum);
            }

            carry_buf[b] = 1.0f / (1.0f + expf(-logits1));
            memcpy(hidden_buf + (int64_t)b * H, h_new_buf, (size_t)H * sizeof(float));

            float* out_t = output->data + base * 2;
            out_t[0] = logits0; out_t[1] = logits1;
        }
    }

    return CE_OK;
}

/* ---- Forward (training: saves state for backward) ---- */
ce_status_t ce_recurrent_expert_forward(ce_module_t* mod,
                                         ce_tensor_t* input,
                                         ce_tensor_t* output) {
    if (!mod || !input || !output) return CE_ERR_INVALID_ARG;
    ce_recurrent_expert_t* st = (ce_recurrent_expert_t*)mod->internal_state;
    if (!st) return CE_ERR_NOT_INITIALIZED;

    int B = (int)input->shape[0];
    int T = (int)input->shape[1];
    int H = st->hidden_size;
    int in_dim = 6 + H;

    /* Release old saved state */
    if (st->saved_inputs) ce_tensor_release(st->saved_inputs);
    if (st->saved_hiddens) ce_tensor_release(st->saved_hiddens);
    if (st->saved_carry) ce_tensor_release(st->saved_carry);
    if (st->saved_logits) ce_tensor_release(st->saved_logits);

    int64_t si_s[3] = {B, T, in_dim};
    int64_t sh_s[3] = {B, T + 1, H};
    int64_t sc_s[2] = {B, T};
    int64_t sl_s[3] = {B, T, 2};
    st->saved_inputs   = ce_tensor_create(si_s, 3, CE_FLOAT32);
    st->saved_hiddens  = ce_tensor_create(sh_s, 3, CE_FLOAT32);
    st->saved_carry    = ce_tensor_create(sc_s, 2, CE_FLOAT32);
    st->saved_logits   = ce_tensor_create(sl_s, 3, CE_FLOAT32);

    float* hidden = st->saved_hiddens->data;
    float* carry_data = st->saved_carry->data;

    /* Pre-allocate per-thread buffers for OpenMP batch parallelism */
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif
    float* elu1_all = (float*)malloc((size_t)n_threads * H * sizeof(float));
    float* elu2_all = (float*)malloc((size_t)n_threads * H * sizeof(float));

    for (int t = 0; t < T; t++) {
        #pragma omp parallel for schedule(dynamic) if(B >= 2)
        for (int b = 0; b < B; b++) {
            int tid = 0;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            #endif
            float* elu1_buf = elu1_all + (size_t)tid * H;
            float* elu2_buf = elu2_all + (size_t)tid * H;

            int64_t base = (int64_t)b * T + t;
            const float* inp_t = input->data + base * 6;

            /* carry_in: 0 for first step, sigmoid(carry_logit) from prev step */
            float carry_in = (t == 0) ? 0.0f : carry_data[(int64_t)b * T + (t - 1)];

            /* Build full input: [a_bit, b_bit, carry_in, op_1, op_2, hidden...] */
            float* si = st->saved_inputs->data + base * in_dim;
            si[0] = inp_t[0]; si[1] = inp_t[1];
            si[2] = carry_in;
            si[3] = inp_t[3]; si[4] = inp_t[4]; si[5] = inp_t[5];
            const float* h_prev = hidden + (int64_t)t * B * H + (int64_t)b * H;
            memcpy(si + 6, h_prev, (size_t)H * sizeof(float));

            /* Layer 1: elu1 = ELU(W1 * x + b1) */
            for (int j = 0; j < H; j++) {
                float sum = st->b1->data[j];
                for (int k = 0; k < in_dim; k++)
                    sum += st->w1->data[j * in_dim + k] * si[k];
                elu1_buf[j] = (sum > 0.0f) ? sum : expf(sum) - 1.0f;
            }

            /* Layer 2: elu2 = ELU(W2 * elu1 + b2) */
            for (int j = 0; j < H; j++) {
                float sum = st->b2->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w2->data[j * H + k] * elu1_buf[k];
                elu2_buf[j] = (sum > 0.0f) ? sum : expf(sum) - 1.0f;
            }

            /* Output: logits = W_out * elu2 + b_out */
            float* sl = st->saved_logits->data + base * 2;
            for (int j = 0; j < 2; j++) {
                float sum = st->b_out->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w_out->data[j * H + k] * elu2_buf[k];
                sl[j] = sum;
            }

            /* Hidden: new_h = tanh(W_h * elu2 + b_h) */
            float* h_new = hidden + (int64_t)(t + 1) * B * H + (int64_t)b * H;
            for (int j = 0; j < H; j++) {
                float sum = st->b_h->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w_h->data[j * H + k] * elu2_buf[k];
                h_new[j] = tanhf(sum);
            }

            /* Carry: sigmoid(carry_logit) */
            carry_data[base] = 1.0f / (1.0f + expf(-sl[1]));

            /* Write output */
            float* out_t = output->data + base * 2;
            out_t[0] = sl[0]; out_t[1] = sl[1];
        }

        /* ── Debug: log activations for first batch element at this timestep ── */
        if (ce_get_config()->enable_debug_logging && B > 0) {
            int db = 0;
            int64_t d_base = (int64_t)db * T + t;
            const float* d_x5h = st->saved_inputs->data + d_base * in_dim;
            const float* d_sl = st->saved_logits->data + d_base * 2;

            /* Recompute elu1/elu2 stats from saved input for logging */
            float d_elu1_buf[H], d_elu2_buf[H];
            memset(d_elu1_buf, 0, sizeof(float) * (size_t)H);
            memset(d_elu2_buf, 0, sizeof(float) * (size_t)H);
            for (int j = 0; j < H; j++) {
                float s = st->b1->data[j];
                for (int k = 0; k < in_dim; k++)
                    s += st->w1->data[j * in_dim + k] * d_x5h[k];
                d_elu1_buf[j] = (s > 0.0f) ? s : expf(s) - 1.0f;
            }
            for (int j = 0; j < H; j++) {
                float s = st->b2->data[j];
                for (int k = 0; k < H; k++)
                    s += st->w2->data[j * H + k] * d_elu1_buf[k];
                d_elu2_buf[j] = (s > 0.0f) ? s : expf(s) - 1.0f;
            }

            fprintf(stderr, "[RecurrentFwd] t=%d b=%d  in=[%.4f %.4f carry=%.4f op=%d,%d]\n",
                    t, db, d_x5h[0], d_x5h[1], d_x5h[2],
                    (int)d_x5h[3], (int)d_x5h[4]);
            fprintf(stderr, "  elu1: mean=%.4f std=%.4f min=%.4f max=%.4f\n",
                    dbg_mean(d_elu1_buf, H), dbg_std(d_elu1_buf, H, dbg_mean(d_elu1_buf, H)),
                    dbg_min(d_elu1_buf, H), dbg_max(d_elu1_buf, H));
            fprintf(stderr, "  elu2: mean=%.4f std=%.4f min=%.4f max=%.4f\n",
                    dbg_mean(d_elu2_buf, H), dbg_std(d_elu2_buf, H, dbg_mean(d_elu2_buf, H)),
                    dbg_min(d_elu2_buf, H), dbg_max(d_elu2_buf, H));
            fprintf(stderr, "  logits: [%.4f, %.4f]  carry=%.4f\n",
                    d_sl[0], d_sl[1], st->saved_carry->data[d_base]);
        }
    }

    free(elu1_all);
    free(elu2_all);

    return CE_OK;
}

/* ---- Backward (BPTT) ---- */
ce_status_t ce_recurrent_expert_backward(ce_module_t* mod,
                                          ce_tensor_t* grad_output,
                                          ce_tensor_t* grad_input) {
    if (!mod || !grad_output) return CE_ERR_INVALID_ARG;
    ce_recurrent_expert_t* st = (ce_recurrent_expert_t*)mod->internal_state;
    if (!st || !st->saved_inputs) return CE_ERR_NOT_INITIALIZED;

    int B = (int)grad_output->shape[0];
    int T = (int)grad_output->shape[1];
    int H = st->hidden_size;
    int in_dim = 6 + H;

    /* Zero all gradients */
    ce_tensor_zero(st->gw1); ce_tensor_zero(st->gb1);
    ce_tensor_zero(st->gw2); ce_tensor_zero(st->gb2);
    ce_tensor_zero(st->gw_out); ce_tensor_zero(st->gb_out);
    ce_tensor_zero(st->gw_h); ce_tensor_zero(st->gb_h);

    float* grad_h_next = (float*)calloc((size_t)B * H, sizeof(float));
    float* grad_x5h = (float*)malloc((size_t)in_dim * sizeof(float));
    float* d_pre_elu2 = (float*)malloc((size_t)H * sizeof(float));
    float* d_elu1 = (float*)malloc((size_t)H * sizeof(float));
    float* d_pre_elu1 = (float*)malloc((size_t)H * sizeof(float));

    /* Accumulated carry gradient: [B, T] — carries the gradient from timestep
     * t+1 back to logits[t][1] through the sigmoid connection.
     * Initialized to zero; accumulated during the backward loop. */
    float* carry_grad_acc = (float*)calloc((size_t)B * T, sizeof(float));

    for (int t = T - 1; t >= 0; t--) {
        for (int b = 0; b < B; b++) {
            /* At the last timestep, no gradient flows from the future */
            if (t == T - 1) {
                for (int j = 0; j < H; j++)
                    grad_h_next[b * H + j] = 0.0f;
            }

            int64_t base = (int64_t)b * T + t;
            const float* x5h = st->saved_inputs->data + base * in_dim;
            const float* h_prev = st->saved_hiddens->data + (int64_t)t * B * H + (int64_t)b * H;
            const float* h_next = st->saved_hiddens->data + (int64_t)(t + 1) * B * H + (int64_t)b * H;

            /* Recompute elu1, elu2, tanh_pre from saved input + hidden */
            float elu1_buf[H], elu2_buf[H], tanh_pre_buf[H];
            for (int j = 0; j < H; j++) {
                float sum = st->b1->data[j];
                for (int k = 0; k < in_dim; k++)
                    sum += st->w1->data[j * in_dim + k] * x5h[k];
                elu1_buf[j] = (sum > 0.0f) ? sum : expf(sum) - 1.0f;
            }
            for (int j = 0; j < H; j++) {
                float sum = st->b2->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w2->data[j * H + k] * elu1_buf[k];
                elu2_buf[j] = (sum > 0.0f) ? sum : expf(sum) - 1.0f;
            }
            for (int j = 0; j < H; j++) {
                float sum = st->b_h->data[j];
                for (int k = 0; k < H; k++)
                    sum += st->w_h->data[j * H + k] * elu2_buf[k];
                tanh_pre_buf[j] = sum;
            }

            float gl[2] = {
                grad_output->data[base * 2],
                grad_output->data[base * 2 + 1]
            };

            /* Add accumulated carry gradient from future timesteps to gl[1].
             * This is the gradient that flows from the loss at t+1 through
             * the carry connection back to the logits at timestep t. */
            gl[1] += carry_grad_acc[b * T + t];

            /* Hidden head backward: dh_pre = grad_h_next * (1 - tanh(z)^2) */
            float dh_pre[H];
            for (int j = 0; j < H; j++) {
                float tv = tanhf(tanh_pre_buf[j]);
                dh_pre[j] = grad_h_next[b * H + j] * (1.0f - tv * tv);
            }

            for (int i = 0; i < H; i++) {
                for (int j = 0; j < H; j++)
                    st->gw_h->data[i * H + j] += dh_pre[i] * elu2_buf[j];
                st->gb_h->data[i] += dh_pre[i];
            }

            /* Output head backward */
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < H; j++)
                    st->gw_out->data[i * H + j] += gl[i] * elu2_buf[j];
                st->gb_out->data[i] += gl[i];
            }

            /* d_elu2 = W_out^T @ gl + W_h^T @ dh_pre */
            float d_elu2[H];
            for (int j = 0; j < H; j++) {
                float sum = 0.0f;
                for (int i = 0; i < 2; i++)
                    sum += st->w_out->data[i * H + j] * gl[i];
                for (int i = 0; i < H; i++)
                    sum += st->w_h->data[i * H + j] * dh_pre[i];
                d_elu2[j] = sum;
            }

            for (int j = 0; j < H; j++)
                d_pre_elu2[j] = d_elu2[j] * ((elu2_buf[j] > 0.0f) ? 1.0f : elu2_buf[j] + 1.0f);

            for (int j = 0; j < H; j++) {
                for (int k = 0; k < H; k++)
                    st->gw2->data[j * H + k] += d_pre_elu2[j] * elu1_buf[k];
                st->gb2->data[j] += d_pre_elu2[j];
            }

            for (int j = 0; j < H; j++) {
                float sum = 0.0f;
                for (int i = 0; i < H; i++)
                    sum += st->w2->data[i * H + j] * d_pre_elu2[i];
                d_elu1[j] = sum;
            }

            for (int j = 0; j < H; j++)
                d_pre_elu1[j] = d_elu1[j] * ((elu1_buf[j] > 0.0f) ? 1.0f : elu1_buf[j] + 1.0f);

            for (int j = 0; j < H; j++) {
                for (int k = 0; k < in_dim; k++)
                    st->gw1->data[j * in_dim + k] += d_pre_elu1[j] * x5h[k];
                st->gb1->data[j] += d_pre_elu1[j];
            }

            for (int k = 0; k < in_dim; k++) {
                float sum = 0.0f;
                for (int j = 0; j < H; j++)
                    sum += st->w1->data[j * in_dim + k] * d_pre_elu1[j];
                grad_x5h[k] = sum;
            }

            for (int j = 0; j < H; j++)
                grad_h_next[b * H + j] = grad_x5h[6 + j];

            /* ── Carry gradient propagation (critical BPTT fix) ──────────
             * carry[t] = sigmoid(logits[t][1])
             * The loss at timestep t depends on carry[t-1] through the
             * network input x6h[2].  grad_x5h[2] holds d_loss_t/d_carry[t-1].
             * We must propagate this back to logits[t-1][1] through the
             * sigmoid derivative: d_logits/d_carry = carry * (1 - carry).
             */
            if (t > 0) {
                int64_t prev_base = (int64_t)b * T + (t - 1);
                float prev_carry = st->saved_carry->data[prev_base];
                float d_logits_prev_1 = grad_x5h[2] * prev_carry * (1.0f - prev_carry);
                carry_grad_acc[prev_base] += d_logits_prev_1;
            }

            /* ── Debug: log gradient norms for first batch element ── */
            if (ce_get_config()->enable_debug_logging && b == 0) {
                fprintf(stderr, "[RecurrentBwd] t=%d b=%d  grad_out=[%.4f, %.4f]  "
                        "grad_hidden_norm=%.4f  grad_gw1=%.4f  grad_gw2=%.4f  "
                        "grad_gw_out=%.4f  grad_gw_h=%.4f\n",
                        t, b, gl[0], gl[1],
                        dbg_norm(grad_h_next, H),
                        dbg_norm(st->gw1->data, st->gw1->numel),
                        dbg_norm(st->gw2->data, st->gw2->numel),
                        dbg_norm(st->gw_out->data, st->gw_out->numel),
                        dbg_norm(st->gw_h->data, st->gw_h->numel));
            }
        }
    }

    if (grad_input) ce_tensor_zero(grad_input);

    free(grad_h_next); free(grad_x5h);
    free(d_pre_elu2); free(d_elu1); free(d_pre_elu1);
    free(carry_grad_acc);
    return CE_OK;
}

/* ---- MoE routing: process batch split by expert ----
 * Routes each sequence to its assigned expert based on op_code.
 * Avoids heap corruption by not creating temporary sub-batch tensors. */
ce_status_t ce_recurrent_expert_forward_routed(ce_module_t** experts,
                                                 int num_experts,
                                                 ce_tensor_t* input,
                                                 ce_tensor_t* output,
                                                 int* assignments /* [B] */) {
    if (!experts || !input || !output || !assignments) return CE_ERR_INVALID_ARG;

    int B = (int)input->shape[0];
    int T = (int)input->shape[1];

    /* Process each sequence individually through its assigned expert.
     * This avoids creating sub-batch tensors (which caused heap corruption).
     * Trade-off: less parallelism, but correct. */
    for (int b = 0; b < B; b++) {
        int eid = assignments[b];
        if (eid < 0 || eid >= num_experts) eid = 0;

        /* Create single-sample input: [1, T, 6] */
        int64_t ss_in_s[3] = {1, T, 6};
        ce_tensor_t* ss_in = ce_tensor_create(ss_in_s, 3, CE_FLOAT32);

        /* Copy this sample's data */
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < 6; f++) {
                ss_in->data[t * 6 + f] = input->data[((int64_t)b * T + t) * 6 + f];
            }
        }

        /* Create single-sample output: [1, T, 2] */
        int64_t ss_out_s[3] = {1, T, 2};
        ce_tensor_t* ss_out = ce_tensor_create(ss_out_s, 3, CE_FLOAT32);

        /* Forward through assigned expert */
        ce_status_t st = ce_recurrent_expert_forward(experts[eid], ss_in, ss_out);
        if (st != CE_OK) {
            ce_tensor_release(ss_in);
            ce_tensor_release(ss_out);
            return st;
        }

        /* Copy results back to batch output */
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < 2; f++) {
                output->data[((int64_t)b * T + t) * 2 + f] = ss_out->data[t * 2 + f];
            }
        }

        ce_tensor_release(ss_in);
        ce_tensor_release(ss_out);
    }

    return CE_OK;
}
