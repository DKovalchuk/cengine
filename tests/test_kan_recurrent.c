/*
 * test_kan_recurrent.c — data-race regression for ce_kan_recurrent_forward.
 *
 * The forward loop runs "#pragma omp parallel for" over the batch, but
 * all per-element scratch buffers (x_t, h1_raw/h1_res/h1_n,
 * h2_raw/h2_res/h2_n, y3, basis_buf, silu_buf) were allocated once
 * before the parallel region and shared by every thread — a data race
 * that corrupts outputs nondeterministically.
 *
 * Regression check: per-element computation is independent, so a serial
 * run (1 thread) and repeated parallel runs (8 threads) over the same
 * input must produce bit-identical outputs.
 */
#include "cengine.h"
#include "memory.h"
#include "tensor.h"
#include "kan_recurrent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  TEST %-46s ", name); \
    tests_run++; \
} while(0)

#define PASS() do { tests_passed++; printf("[PASS]\n"); } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); } while(0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

/* Deterministic pseudo-random fill in [0,1] (LCG, fixed seed). */
static void fill_input(ce_tensor_t* t) {
    uint32_t s = 12345u;
    for (int64_t i = 0; i < t->numel; i++) {
        s = s * 1664525u + 1013904223u;
        t->data[i] = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
    }
}

static float max_abs_diff(const ce_tensor_t* a, const ce_tensor_t* b) {
    float maxd = 0.0f;
    for (int64_t i = 0; i < a->numel; i++) {
        float d = fabsf(a->data[i] - b->data[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

static void test_parallel_matches_serial(void) {
    TEST("kan_recurrent: parallel == serial (bitwise)");
    const int B = 16, T = 8, H = 16;

    ce_module_t* mod = ce_kan_recurrent_create(H, "kr_race");
    ASSERT(mod != NULL, "create failed");

    int64_t in_s[3]  = {B, T, 5};
    int64_t out_s[3] = {B, T, 2};
    ce_tensor_t* in     = ce_tensor_zeros(in_s, 3);
    ce_tensor_t* out_p  = ce_tensor_zeros(out_s, 3);
    ce_tensor_t* out_ref = ce_tensor_zeros(out_s, 3);
    ASSERT(in && out_p && out_ref, "tensor alloc failed");
    fill_input(in);

    /* Serial reference: single thread, run twice to prove determinism. */
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    ASSERT(ce_kan_recurrent_forward(mod, in, out_ref) == CE_OK,
           "serial forward #1 failed");
    ASSERT(ce_kan_recurrent_forward(mod, in, out_p) == CE_OK,
           "serial forward #2 failed");
    ASSERT(memcmp(out_ref->data, out_p->data,
                  (size_t)out_ref->numel * sizeof(float)) == 0,
           "serial forward is not reproducible");

    /* Parallel runs must match the serial reference bit-for-bit:
     * each batch element's computation is independent, so any mismatch
     * means threads interfered through shared scratch buffers. */
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    for (int run = 0; run < 10; run++) {
        ASSERT(ce_kan_recurrent_forward(mod, in, out_p) == CE_OK,
               "parallel forward failed");
        if (memcmp(out_ref->data, out_p->data,
                   (size_t)out_ref->numel * sizeof(float)) != 0) {
            char buf[160];
            snprintf(buf, sizeof buf,
                     "run %d diverges from serial: max|diff|=%g (data race)",
                     run, (double)max_abs_diff(out_ref, out_p));
            FAIL(buf);
            return;
        }
    }

    ce_tensor_release(in);
    ce_tensor_release(out_p);
    ce_tensor_release(out_ref);
    /* Module intentionally not destroyed: test process exits right after. */
    PASS();
}

/* ---- SiLU reference test (major C2) ----
 *
 * kan_forward_1d_avx2 computed SiLU as x*sigmoid(-x) (both the AVX2
 * loop and the scalar tail), while the intended activation — per the
 * code's own "SiLU" comments — is x*sigmoid(x).  The reference below
 * mirrors the B=1, T=1 forward with libm expf: kan1 (in_f=5 → scalar
 * tail site), kan2/kan3 (in_f=8 → AVX2 site) — both bug sites are
 * exercised.
 */
static float ref_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void lcg_fill(float* p, int64_t n, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
    }
}

/* KAN apply as kan_forward_1d_avx2 computes it, with TRUE SiLU. */
static void kan_apply_ref(const float* x,
                          const ce_tensor_t* w_b, const ce_tensor_t* w_s,
                          const ce_tensor_t* c, const ce_tensor_t* bias,
                          int in_f, int out_f, float* y) {
    float* basis = (float*)malloc((size_t)in_f * 4 * sizeof(float));
    float* silu  = (float*)malloc((size_t)in_f * sizeof(float));
    for (int i = 0; i < in_f; i++) {
        float xi = x[i];
        basis[i * 4 + 0] = 1.0f;
        basis[i * 4 + 1] = xi;
        basis[i * 4 + 2] = xi * xi;
        basis[i * 4 + 3] = xi * xi * xi;
        silu[i] = xi * ref_sigmoid(xi);
    }
    for (int j = 0; j < out_f; j++) {
        float v = bias->data[j];
        for (int i = 0; i < in_f; i++) {
            float sv = 0.0f;
            for (int k = 0; k < 4; k++)
                sv += c->data[(j * in_f + i) * 4 + k] * basis[i * 4 + k];
            v += w_b->data[j * in_f + i] * silu[i]
               + w_s->data[j * in_f + i] * sv;
        }
        y[j] = v;
    }
    free(basis);
    free(silu);
}

/* LayerNorm as layer_norm_1d computes it (gamma/beta from tensors). */
static void ln_ref(const float* x, float* y, int H,
                   const ce_tensor_t* gamma, const ce_tensor_t* beta) {
    float mean = 0.0f;
    for (int i = 0; i < H; i++) mean += x[i];
    mean /= (float)H;
    float var = 0.0f;
    for (int i = 0; i < H; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float std_val = sqrtf(var / (float)H + 1e-5f);
    for (int i = 0; i < H; i++)
        y[i] = gamma->data[i] * (x[i] - mean) / std_val + beta->data[i];
}

static void test_silu_matches_reference(void) {
    TEST("kan_recurrent: forward matches TRUE SiLU ref");
    const int B = 1, T = 1, H = 8, in_f = 5;

    ce_module_t* mod = ce_kan_recurrent_create(H, "kr_silu");
    ASSERT(mod != NULL, "create failed");

    /* Parameter add order in create(): res_proj, ln1_g, ln1_b, ln2_g,
     * ln2_b, pos_emb, then kan1(w_b,w_s,c,bias), kan2, kan3.  The KAN
     * sub-params are xavier-seeded from a pointer value — overwrite
     * everything that feeds the output with a deterministic stream. */
    ASSERT(ce_module_param_count(mod) == 18, "unexpected param count");
    ce_tensor_t* res_proj = ce_module_get_parameter(mod, 0);
    ce_tensor_t* ln1_g    = ce_module_get_parameter(mod, 1);
    ce_tensor_t* ln1_b    = ce_module_get_parameter(mod, 2);
    ce_tensor_t* ln2_g    = ce_module_get_parameter(mod, 3);
    ce_tensor_t* ln2_b    = ce_module_get_parameter(mod, 4);
    ce_tensor_t* pos_emb  = ce_module_get_parameter(mod, 5);
    ce_tensor_t* k1_wb = ce_module_get_parameter(mod, 6);
    ce_tensor_t* k1_ws = ce_module_get_parameter(mod, 7);
    ce_tensor_t* k1_c  = ce_module_get_parameter(mod, 8);
    ce_tensor_t* k1_b  = ce_module_get_parameter(mod, 9);
    ce_tensor_t* k2_wb = ce_module_get_parameter(mod, 10);
    ce_tensor_t* k2_ws = ce_module_get_parameter(mod, 11);
    ce_tensor_t* k2_c  = ce_module_get_parameter(mod, 12);
    ce_tensor_t* k2_b  = ce_module_get_parameter(mod, 13);
    ce_tensor_t* k3_wb = ce_module_get_parameter(mod, 14);
    ce_tensor_t* k3_ws = ce_module_get_parameter(mod, 15);
    ce_tensor_t* k3_c  = ce_module_get_parameter(mod, 16);
    ce_tensor_t* k3_b  = ce_module_get_parameter(mod, 17);
    ASSERT(res_proj && ln1_g && ln1_b && ln2_g && ln2_b && pos_emb &&
           k1_wb && k1_ws && k1_c && k1_b && k2_wb && k2_ws && k2_c &&
           k2_b && k3_wb && k3_ws && k3_c && k3_b, "get_parameter failed");

    lcg_fill(res_proj->data, res_proj->numel, 111u);
    lcg_fill(k1_wb->data, k1_wb->numel, 222u);
    lcg_fill(k1_ws->data, k1_ws->numel, 333u);
    lcg_fill(k1_c->data,  k1_c->numel,  444u);
    lcg_fill(k1_b->data,  k1_b->numel,  555u);
    lcg_fill(k2_wb->data, k2_wb->numel, 666u);
    lcg_fill(k2_ws->data, k2_ws->numel, 777u);
    lcg_fill(k2_c->data,  k2_c->numel,  888u);
    lcg_fill(k2_b->data,  k2_b->numel,  999u);
    lcg_fill(k3_wb->data, k3_wb->numel, 1234u);
    lcg_fill(k3_ws->data, k3_ws->numel, 2345u);
    lcg_fill(k3_c->data,  k3_c->numel,  3456u);
    lcg_fill(k3_b->data,  k3_b->numel,  4567u);

    float inp[5];
    lcg_fill(inp, 5, 9876u);
    int64_t in_s[3] = {B, T, in_f};
    int64_t out_s[3] = {B, T, 2};
    ce_tensor_t* in  = ce_tensor_create_from_data(inp, in_s, 3);
    ce_tensor_t* out = ce_tensor_zeros(out_s, 3);
    ASSERT(in && out, "tensor alloc");

    ASSERT(ce_kan_recurrent_forward(mod, in, out) == CE_OK,
           "forward failed");

    /* Reference for B=1, T=1 (carry == 0 at t=0). */
    float x[5];
    for (int i = 0; i < in_f; i++) x[i] = inp[i] * 2.0f - 1.0f;
    x[2] = 0.0f * 2.0f - 1.0f;                      /* carry == 0 */
    for (int i = 0; i < in_f; i++) x[i] += pos_emb->data[0 * in_f + i];

    float h1_raw[H], h1_res[H], h1_n[H];
    kan_apply_ref(x, k1_wb, k1_ws, k1_c, k1_b, in_f, H, h1_raw);
    for (int j = 0; j < H; j++) {
        float proj = 0.0f;
        for (int i = 0; i < in_f; i++)
            proj += res_proj->data[j * in_f + i] * x[i];
        h1_res[j] = h1_raw[j] + proj * 0.1f;
    }
    ln_ref(h1_res, h1_n, H, ln1_g, ln1_b);

    float h2_raw[H], h2_res[H], h2_n[H];
    kan_apply_ref(h1_n, k2_wb, k2_ws, k2_c, k2_b, H, H, h2_raw);
    for (int j = 0; j < H; j++)
        h2_res[j] = h2_raw[j] + h1_n[j];
    ln_ref(h2_res, h2_n, H, ln2_g, ln2_b);

    float y3[2];
    kan_apply_ref(h2_n, k3_wb, k3_ws, k3_c, k3_b, H, 2, y3);

    float maxd = 0.0f;
    for (int64_t i = 0; i < 2; i++) {
        float dd = fabsf(out->data[i] - y3[i]);
        if (dd > maxd) maxd = dd;
    }
    if (maxd > 5e-3f) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "max|diff|=%g — forward does not match true SiLU "
                 "(x*sigmoid(-x) vs x*sigmoid(x)), want <0.005",
                 (double)maxd);
        FAIL(buf);
        return;
    }

    ce_tensor_release(in);
    ce_tensor_release(out);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Recurrent-KAN Tests ===\n\n");
    ce_init();

    test_parallel_matches_serial();
    test_silu_matches_reference();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
