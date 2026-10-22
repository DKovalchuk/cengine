/*
 * test_kan_layer.c — KAN layer SiLU / output-scatter regression tests.
 *
 * Regression for a major bug (C1): the AVX2 forward/backward computed
 * SiLU as x * sigmoid(-x) (avx2_sigmoid_ps(neg_x)) while the scalar tail
 * used the correct x * sigmoid(x) — so results depended on whether
 * in_features crossed the 8-wide AVX2 boundary.  The AVX2 output stage
 * also mixed lane layouts between the duplicated-pair weights (wb/ws)
 * and the sequential-repeated coefficients (sv), so y[j+1..j+3] received
 * wrong phi values and phi for j+3 was dropped entirely.
 *
 * Reference values are computed with libm expf — approximation noise of
 * the fast exp paths is orders of magnitude below the bug signal.
 *
 * Backward is checked against a non-trivial grad_output pattern, which
 * pins both the d_silu term and the chain-rule grad_output factor in
 * grad_input.
 */
#include "cengine.h"
#include "kan_layer.h"
#include "memory.h"
#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  TEST %-52s ", name); \
    tests_run++; \
} while(0)

#define PASS() do { tests_passed++; printf("[PASS]\n"); } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); } while(0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

/* Tolerance: fast-exp approximation noise is ≲1e-4, the σ(-x)/scatter
 * bugs produce O(1e-2..1) errors. */
#define TOL 5e-3f

static float ref_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Deterministic pseudo-random fill in [-0.5, 0.5]. */
static void lcg_fill(float* p, int64_t n, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
    }
}

/* Give the layer fully deterministic parameters (create() seeds w_s from
 * a pointer value, which is not reproducible). */
static void set_deterministic_params(ce_module_t* mod) {
    ce_kan_layer_t* st = (ce_kan_layer_t*)mod->internal_state;
    int of = st->out_features, inf = st->in_features;
    lcg_fill(st->w_b->data, (int64_t)of * inf, 1001u);
    lcg_fill(st->w_s->data, (int64_t)of * inf, 2002u);
    lcg_fill(st->c->data, (int64_t)of * inf * 4, 3003u);
    lcg_fill(st->bias->data, of, 4004u);
}

/* Scalar reference forward for input [1,1,in_f] → y[out_f]:
 *   xn    = x*2 - 1                        (layer normalises input)
 *   silu  = xn * sigmoid(xn)               (true SiLU)
 *   basis = [1, x, x^2, x^3]
 *   y[j]  = bias[j] + sum_i w_b*silu + w_s*spline */
static void ref_forward(const ce_module_t* mod, const float* inp,
                        float* y, float* xn_out, float* silu_out,
                        float* basis_out) {
    const ce_kan_layer_t* st = (const ce_kan_layer_t*)mod->internal_state;
    int inf = st->in_features, of = st->out_features;

    for (int i = 0; i < inf; i++) {
        float x = inp[i] * 2.0f - 1.0f;
        xn_out[i] = x;
        silu_out[i] = x * ref_sigmoid(x);
        basis_out[i * 4 + 0] = 1.0f;
        basis_out[i * 4 + 1] = x;
        basis_out[i * 4 + 2] = x * x;
        basis_out[i * 4 + 3] = x * x * x;
    }
    for (int j = 0; j < of; j++) {
        float v = st->bias->data[j];
        for (int i = 0; i < inf; i++) {
            float sv = 0.0f;
            for (int k = 0; k < 4; k++)
                sv += st->c->data[(j * inf + i) * 4 + k] * basis_out[i * 4 + k];
            v += st->w_b->data[j * inf + i] * silu_out[i]
               + st->w_s->data[j * inf + i] * sv;
        }
        y[j] = v;
    }
}

static float max_diff(const float* a, const float* b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ---- Forward: AVX2 SiLU (in_f=8) + AVX2 scatter + scalar tail (out_f=5) */
static void test_forward_avx2_matches_reference(void) {
    TEST("kan_layer forward: AVX2 in_f=8, out_f=5 vs scalar ref");
    const int in_f = 8, out_f = 5;

    ce_module_t* mod = ce_kan_layer_create(in_f, out_f, "kl_avx2");
    ASSERT(mod != NULL, "create failed");
    set_deterministic_params(mod);

    float inp[8];
    lcg_fill(inp, 8, 7777u);
    int64_t in_s[3] = {1, 1, in_f};
    int64_t out_s[3] = {1, 1, out_f};
    ce_tensor_t* in = ce_tensor_create_from_data(inp, in_s, 3);
    ce_tensor_t* out = ce_tensor_zeros(out_s, 3);
    ASSERT(in && out, "tensor alloc");

    ASSERT(ce_kan_layer_forward(mod, in, out) == CE_OK, "forward failed");

    float ref[8], xn[8], silu[8], basis[8 * 4];
    ref_forward(mod, inp, ref, xn, silu, basis);

    float d = max_diff(out->data, ref, out_f);
    if (d > TOL) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "max|diff|=%g (SiLU/scatter divergence, want <%g)",
                 (double)d, (double)TOL);
        FAIL(buf);
        return;
    }

    ce_tensor_release(in);
    ce_tensor_release(out);
    ce_module_destroy(mod);
    PASS();
}

/* ---- Forward: scalar SiLU (in_f=3) but AVX2 output scatter (out_f=4)
 *       — isolates the phi_arr lane-layout bug from the SiLU bug. */
static void test_forward_scatter_matches_reference(void) {
    TEST("kan_layer forward: scalar in_f=3, scatter out_f=4 vs ref");
    const int in_f = 3, out_f = 4;

    ce_module_t* mod = ce_kan_layer_create(in_f, out_f, "kl_scatter");
    ASSERT(mod != NULL, "create failed");
    set_deterministic_params(mod);

    float inp[3];
    lcg_fill(inp, 3, 5555u);
    int64_t in_s[3] = {1, 1, in_f};
    int64_t out_s[3] = {1, 1, out_f};
    ce_tensor_t* in = ce_tensor_create_from_data(inp, in_s, 3);
    ce_tensor_t* out = ce_tensor_zeros(out_s, 3);
    ASSERT(in && out, "tensor alloc");

    ASSERT(ce_kan_layer_forward(mod, in, out) == CE_OK, "forward failed");

    float ref[4], xn[3], silu[3], basis[3 * 4];
    ref_forward(mod, inp, ref, xn, silu, basis);

    float d = max_diff(out->data, ref, out_f);
    if (d > TOL) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "max|diff|=%g — AVX2 output scatter mixes phi lanes "
                 "(want <%g)", (double)d, (double)TOL);
        FAIL(buf);
        return;
    }

    ce_tensor_release(in);
    ce_tensor_release(out);
    ce_module_destroy(mod);
    PASS();
}

/* ---- Backward: d_silu uses sigmoid(+x); grad_input is weighted by the
 *       upstream grad_output (chain rule); g_w_b = g_j * silu ------- */
static void test_backward_matches_reference(void) {
    TEST("kan_layer backward: g_out-weighted grad vs scalar ref");
    const int in_f = 8, out_f = 5;

    ce_module_t* mod = ce_kan_layer_create(in_f, out_f, "kl_bw");
    ASSERT(mod != NULL, "create failed");
    set_deterministic_params(mod);
    ce_kan_layer_t* st = (ce_kan_layer_t*)mod->internal_state;

    float inp[8];
    lcg_fill(inp, 8, 4242u);
    float g_pat[8];
    lcg_fill(g_pat, 8, 6868u);
    for (int j = 0; j < out_f; j++)
        g_pat[j] = g_pat[j] * 1.6f + 1.1f;   /* spread over [0.3, 1.9] */
    int64_t in_s[3] = {1, 1, in_f};
    int64_t out_s[3] = {1, 1, out_f};
    ce_tensor_t* in = ce_tensor_create_from_data(inp, in_s, 3);
    ce_tensor_t* out = ce_tensor_zeros(out_s, 3);
    ce_tensor_t* g_out = ce_tensor_create_from_data(g_pat, out_s, 3);
    ce_tensor_t* g_in = ce_tensor_zeros(in_s, 3);
    ASSERT(in && out && g_out && g_in, "tensor alloc");

    ASSERT(ce_kan_layer_forward(mod, in, out) == CE_OK, "forward failed");
    ASSERT(ce_kan_layer_backward(mod, g_out, g_in) == CE_OK,
           "backward failed");

    /* Reference with the same deterministic params and xn from forward. */
    float ref[8], xn[8], silu[8], basis[8 * 4];
    ref_forward(mod, inp, ref, xn, silu, basis);

    /* d_silu = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
     * grad_in[i] = 2 * sum_j g_j*(w_b*d_silu + w_s*sum_k c*d_basis_k) —
     * the upstream g_j must weight the sum (chain rule). */
    float grad_ref[8];
    for (int i = 0; i < in_f; i++) {
        float x = xn[i];
        float sig = ref_sigmoid(x);
        float dsilu = sig * (1.0f + x * (1.0f - sig));
        float d_basis[4] = {0.0f, 1.0f, 2.0f * x, 3.0f * x * x};
        float acc = 0.0f;
        for (int j = 0; j < out_f; j++) {
            float d_spline = 0.0f;
            for (int k = 0; k < 4; k++)
                d_spline += st->c->data[(j * in_f + i) * 4 + k] * d_basis[k];
            acc += g_pat[j] *
                 (st->w_b->data[j * in_f + i] * dsilu
                + st->w_s->data[j * in_f + i] * d_spline);
        }
        grad_ref[i] = acc * 2.0f;
    }

    float d = max_diff(g_in->data, grad_ref, in_f);
    if (d > TOL) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "grad_input max|diff|=%g (d_silu uses sigmoid(-x)? "
                 "want <%g)", (double)d, (double)TOL);
        FAIL(buf);
        return;
    }

    /* g_w_b[j,i] = sum_n g_j * silu[n,i]; N=1 → g_pat[j]*silu[i]
     * over the [out_f, in_f] gradient layout. */
    float gwb_ref[8 * 5];
    for (int j = 0; j < out_f; j++)
        for (int i = 0; i < in_f; i++)
            gwb_ref[j * in_f + i] = g_pat[j] * silu[i];
    d = max_diff(st->g_w_b->data, gwb_ref, (int64_t)in_f * out_f);
    if (d > TOL) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "g_w_b max|diff|=%g (saved silu used sigmoid(-x)? "
                 "want <%g)", (double)d, (double)TOL);
        FAIL(buf);
        return;
    }

    ce_tensor_release(in);
    ce_tensor_release(out);
    ce_tensor_release(g_out);
    ce_tensor_release(g_in);
    ce_module_destroy(mod);
    PASS();
}

int main(void) {
    printf("\n=== CEngine KAN Layer Tests ===\n\n");
    ce_init();

    test_forward_avx2_matches_reference();
    test_forward_scatter_matches_reference();
    test_backward_matches_reference();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
