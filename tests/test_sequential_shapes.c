/*
 * test_sequential_shapes.c — sequential intermediate/gradient buffer
 * sizing (major D6).
 *
 * ce_sequential_forward/backward sized their intermediate buffers with
 * a heuristic keyed ONLY on `layer->forward == ce_linear_forward`.
 * A KAN child mid-chain kept the input's shape, so:
 *   forward: inter was [.., in_features] while the layer wrote
 *            out_features values per row → heap out-of-bounds write
 *            whenever out_features > in_features, plus garbage rows;
 *   backward: the gradient buffer kept the output shape instead of
 *            in_features.
 *
 * The chain below is 3D end-to-end (KAN's forward expects [B,T,F]):
 *   KAN(4→3) → KAN(3→6) → KAN(6→1)
 * The middle KAN grows 3→6 against an input-shaped buffer — the OOB.
 * Reference values come from running the children manually with
 * correctly sized tensors.
 */
#include "cengine.h"
#include "kan_layer.h"
#include "layers.h"
#include "tensor.h"
#include <math.h>
#include <stdio.h>
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

static void lcg_fill(float* p, int64_t n, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
    }
}

/* KAN init is xavier-seeded from a pointer value — overwrite everything
 * that feeds the output so the run is reproducible. */
static void set_kan_params(ce_module_t* mod, uint32_t seed) {
    ce_kan_layer_t* st = (ce_kan_layer_t*)mod->internal_state;
    lcg_fill(st->w_b->data, st->w_b->numel, seed);
    lcg_fill(st->w_s->data, st->w_s->numel, seed + 1);
    lcg_fill(st->c->data,  st->c->numel,  seed + 2);
    lcg_fill(st->bias->data, st->bias->numel, seed + 3);
}

static float max_diff(const float* a, const float* b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void test_kan_midchain_shapes(void) {
    TEST("sequential: KAN mid-chain fwd/bwd match manual ref");
    const int B = 1, T = 2;
    const int f0 = 4, f1 = 3, f2 = 6, f3 = 1;

    ce_module_t* k1 = ce_kan_layer_create(f0, f1, "k1");
    ce_module_t* k2 = ce_kan_layer_create(f1, f2, "k2");
    ce_module_t* k3 = ce_kan_layer_create(f2, f3, "k3");
    ASSERT(k1 && k2 && k3, "kan create");
    set_kan_params(k1, 111u);
    set_kan_params(k2, 222u);
    set_kan_params(k3, 333u);

    ce_module_t* seq = ce_sequential_create("seq_kan");
    ASSERT(seq != NULL, "seq create");
    ASSERT(ce_sequential_add(seq, k1) == CE_OK, "add k1");
    ASSERT(ce_sequential_add(seq, k2) == CE_OK, "add k2");
    ASSERT(ce_sequential_add(seq, k3) == CE_OK, "add k3");

    float in_data[B * T * f0];
    lcg_fill(in_data, (int64_t)B * T * f0, 4242u);
    int64_t in_sh[3]  = {B, T, f0};
    int64_t out_sh[3] = {B, T, f3};
    ce_tensor_t* input = ce_tensor_create_from_data(in_data, in_sh, 3);
    ce_tensor_t* out   = ce_tensor_zeros(out_sh, 3);
    ASSERT(input && out, "tensor alloc");

    ce_status_t st = ce_sequential_forward(seq, input, out);
    if (st != CE_OK) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "forward returned 0x%x — mid-chain KAN got an "
                 "input-sized intermediate buffer", st);
        FAIL(buf);
        return;
    }

    /* Manual reference: children in order with correctly sized tensors. */
    int64_t sh1[3] = {B, T, f1};
    int64_t sh2[3] = {B, T, f2};
    ce_tensor_t* t1 = ce_tensor_zeros(sh1, 3);
    ce_tensor_t* t2 = ce_tensor_zeros(sh2, 3);
    ce_tensor_t* ref = ce_tensor_zeros(out_sh, 3);
    ASSERT(t1 && t2 && ref, "ref alloc");
    ASSERT(k1->forward(k1, input, t1) == CE_OK, "manual k1");
    ASSERT(k2->forward(k2, t1, t2) == CE_OK, "manual k2");
    ASSERT(k3->forward(k3, t2, ref) == CE_OK, "manual k3");

    float d = max_diff(out->data, ref->data, out->numel);
    if (d > 1e-5f) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "sequential output differs from manual reference "
                 "max|diff|=%g (wrong mid-chain buffer shape)",
                 (double)d);
        FAIL(buf);
        return;
    }

    /* Backward pin: grad_input must equal the manual chain too. */
    int64_t g_sh[3] = {B, T, f3};
    float g_data[B * T * f3];
    lcg_fill(g_data, (int64_t)B * T * f3, 555u);
    ce_tensor_t* g_out = ce_tensor_create_from_data(g_data, g_sh, 3);
    ce_tensor_t* g_in  = ce_tensor_zeros(in_sh, 3);
    ce_tensor_t* m_g2  = ce_tensor_zeros(sh2, 3);
    ce_tensor_t* m_g1  = ce_tensor_zeros(sh1, 3);
    ce_tensor_t* m_g0  = ce_tensor_zeros(in_sh, 3);
    ASSERT(g_out && g_in && m_g2 && m_g1 && m_g0, "grad alloc");

    st = ce_sequential_backward(seq, g_out, g_in);
    if (st != CE_OK) {
        char buf[160];
        snprintf(buf, sizeof buf, "backward returned 0x%x", st);
        FAIL(buf);
        return;
    }
    ASSERT(k3->backward(k3, g_out, m_g2) == CE_OK, "manual bw k3");
    ASSERT(k2->backward(k2, m_g2, m_g1) == CE_OK, "manual bw k2");
    ASSERT(k1->backward(k1, m_g1, m_g0) == CE_OK, "manual bw k1");

    d = max_diff(g_in->data, m_g0->data, g_in->numel);
    if (d > 1e-4f) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "sequential grad_input differs from manual "
                 "max|diff|=%g", (double)d);
        FAIL(buf);
        return;
    }

    ce_tensor_release(input);
    ce_tensor_release(out);
    ce_tensor_release(t1);
    ce_tensor_release(t2);
    ce_tensor_release(ref);
    ce_tensor_release(g_out);
    ce_tensor_release(g_in);
    ce_tensor_release(m_g2);
    ce_tensor_release(m_g1);
    ce_tensor_release(m_g0);
    ce_module_destroy(seq);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Sequential Shape Tests ===\n\n");
    ce_init();

    test_kan_midchain_shapes();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
