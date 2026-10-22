/*
 * test_reduction_dims.c — dimensional reductions over arbitrary axes
 * (major D7).
 *
 * cpu_sum treated EVERY dim != 1 as "sum over rows of a 2D matrix":
 * correct for dim==0 only — on a 3D tensor with dim==1 it summed the
 * wrong axis (values mixed across dim2) and with dim==2 it memset
 * `cols = numel/shape[0]` floats into the smaller output (heap
 * out-of-bounds) and computed the wrong axis.  cpu_mean inherits cpu_sum.
 * cpu_max_reduce implemented ONLY dim==1 and returned CE_OK for every
 * other dim without writing the output at all (silent no-op), and did
 * not validate `dim` — an out-of-range axis "succeeded".
 *
 * All expectations are computed with a strided reference in the test.
 * Clean-RED tests run before the two heap-corruption ones; stdout is
 * unbuffered so a crash still shows which tests already ran.
 */
#include "cengine.h"
#include "compute.h"
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

/* Generic strided reference: reduce `in` over `dim`. */
static void ref_reduce(const ce_tensor_t* in, int dim, bool want_max,
                       float* out) {
    int64_t outer = 1, inner = 1, r = in->shape[dim];
    for (int d = 0; d < dim; d++) outer *= in->shape[d];
    for (int d = dim + 1; d < in->ndim; d++) inner *= in->shape[d];

    for (int64_t u = 0; u < outer; u++) {
        for (int64_t i = 0; i < inner; i++) {
            float acc = in->data[(u * r) * inner + i];
            for (int64_t k = 1; k < r; k++) {
                float v = in->data[(u * r + k) * inner + i];
                if (want_max) { if (v > acc) acc = v; }
                else acc += v;
            }
            out[u * inner + i] = acc;
        }
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

static void fill_seq(float* p, int64_t n) {
    for (int64_t i = 0; i < n; i++) p[i] = (float)(i % 17) - 8.0f;
}

static ce_tensor_t* make3d(float* data, int64_t a, int64_t b, int64_t c) {
    int64_t sh[3] = {a, b, c};
    return ce_tensor_create_from_data(data, sh, 3);
}

/* ---- pins: behaviour that was already correct must not change ---- */
static void test_legacy_2d_and_3d_dim0(void) {
    TEST("sum: legacy 2D dim0/dim1 and 3D dim0 pins");
    float d2[6] = {1, 2, 3, 4, 5, 6};
    int64_t sh2[2] = {2, 3};
    ce_tensor_t* t2 = ce_tensor_create_from_data(d2, sh2, 2);
    ASSERT(t2, "alloc");

    float o0[3], o1[2], ref[3];
    int64_t s0[1] = {3}, s1[1] = {2};
    ce_tensor_t* out0 = ce_tensor_zeros(s0, 1);
    ce_tensor_t* out1 = ce_tensor_zeros(s1, 1);
    ASSERT(out0 && out1, "alloc");

    ASSERT(ce_get_default_backend()->sum(t2, out0, 0) == CE_OK, "2d dim0");
    ref_reduce(t2, 0, false, ref);
    ASSERT(max_diff(out0->data, ref, 3) < 1e-5f, "2d dim0 values");

    ASSERT(ce_get_default_backend()->sum(t2, out1, 1) == CE_OK, "2d dim1");
    ref_reduce(t2, 1, false, ref);
    ASSERT(max_diff(out1->data, ref, 2) < 1e-5f, "2d dim1 values");

    float d3[24];
    fill_seq(d3, 24);
    ce_tensor_t* t3 = make3d(d3, 2, 3, 4);
    ASSERT(t3, "alloc");
    int64_t sh0[2] = {3, 4};
    ce_tensor_t* o30 = ce_tensor_zeros(sh0, 2);
    ASSERT(o30, "alloc");
    ASSERT(ce_get_default_backend()->sum(t3, o30, 0) == CE_OK, "3d dim0");
    ref_reduce(t3, 0, false, ref);
    ASSERT(max_diff(o30->data, ref, 12) < 1e-5f, "3d dim0 values");

    ce_tensor_release(t2); ce_tensor_release(out0); ce_tensor_release(out1);
    ce_tensor_release(t3); ce_tensor_release(o30);
    PASS();
}

/* ---- RED: 3D dim1 sums the wrong axis (mixes dim2 into the result) */
static void test_3d_sum_dim1(void) {
    TEST("sum: 3D dim1 reduces only dim1");
    float d[24];
    fill_seq(d, 24);
    ce_tensor_t* t = make3d(d, 2, 3, 4);
    ASSERT(t, "alloc");

    int64_t sh[2] = {2, 4};
    ce_tensor_t* out = ce_tensor_zeros(sh, 2);
    ASSERT(out, "alloc");
    ASSERT(ce_get_default_backend()->sum(t, out, 1) == CE_OK, "dim1 sum");

    float ref[8];
    ref_reduce(t, 1, false, ref);
    float md = max_diff(out->data, ref, 8);
    if (md > 1e-4f) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "max|diff|=%g — dim1 must sum only axis1 (got a "
                 "rows-reduction mixing axis2)", (double)md);
        FAIL(buf);
        return;
    }
    ce_tensor_release(t); ce_tensor_release(out);
    PASS();
}

/* ---- RED: max over non-1 dims was a silent no-op (out untouched) ---- */
static void test_3d_max_dims(void) {
    TEST("max: 3D dim0/dim2 reduce + indices (no-op before)");
    float d[24];
    fill_seq(d, 24);
    ce_tensor_t* t = make3d(d, 2, 3, 4);
    ASSERT(t, "alloc");

    int64_t sh0[2] = {3, 4};
    ce_tensor_t* out0 = ce_tensor_zeros(sh0, 2);
    ASSERT(out0, "alloc");
    ASSERT(ce_get_default_backend()->max(t, out0, 0, NULL) == CE_OK,
           "dim0 max");
    float ref[12];
    ref_reduce(t, 0, true, ref);
    float md = max_diff(out0->data, ref, 12);
    if (md > 0.0f) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "max|diff|=%g — dim0 output never written (silent "
                 "no-op for dim != 1)", (double)md);
        FAIL(buf);
        return;
    }

    int64_t sh2[2] = {2, 3};
    ce_tensor_t* out2 = ce_tensor_zeros(sh2, 2);
    ce_tensor_t* idx2 = ce_tensor_zeros(sh2, 2);
    ASSERT(out2 && idx2, "alloc");
    ASSERT(ce_get_default_backend()->max(t, out2, 2, idx2) == CE_OK,
           "dim2 max");
    ref_reduce(t, 2, true, ref);
    md = max_diff(out2->data, ref, 6);
    if (md > 0.0f) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "max|diff|=%g — dim2 max never written (silent no-op)",
                 (double)md);
        FAIL(buf);
        return;
    }
    /* Indices must point at the max inside each dim2 run of 4. */
    for (int64_t u = 0; u < 6; u++) {
        float expect = -1e30f; int64_t best = 0;
        for (int64_t k = 0; k < 4; k++) {
            float v = t->data[u * 4 + k];
            if (v > expect) { expect = v; best = k; }
        }
        ASSERT(idx2->data[u] == (float)best, "indices wrong");
    }

    ce_tensor_release(t); ce_tensor_release(out0);
    ce_tensor_release(out2); ce_tensor_release(idx2);
    PASS();
}

/* ---- RED: out-of-range dim on max_reduce returned CE_OK silently ---- */
static void test_invalid_dims(void) {
    TEST("reductions: out-of-range dim rejected (max returned OK)");
    float d[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int64_t sh[2] = {2, 4};
    ce_tensor_t* t = ce_tensor_create_from_data(d, sh, 2);
    ASSERT(t, "alloc");
    int64_t osh[1] = {2};
    ce_tensor_t* out = ce_tensor_zeros(osh, 1);
    ASSERT(out, "alloc");

    ASSERT(ce_get_default_backend()->sum(t, out, 2) == CE_ERR_INVALID_ARG,
           "sum must reject dim == ndim");
    ce_status_t st = ce_get_default_backend()->max(t, out, 7, NULL);
    if (st == CE_OK) {
        FAIL("max_reduce accepted dim == 7 (no validation)");
        return;
    }
    ASSERT(st == CE_ERR_INVALID_ARG, "max must reject dim == ndim");

    ce_tensor_release(t); ce_tensor_release(out);
    PASS();
}

/* ---- RED: wrong-sized output must be rejected, not written past ---- */
static void test_output_shape_guard(void) {
    TEST("reductions: wrong-sized output rejected (INVALID_SHAPE)");
    float d[6] = {1, 2, 3, 4, 5, 6};
    int64_t sh[2] = {2, 3};
    ce_tensor_t* t = ce_tensor_create_from_data(d, sh, 2);
    ASSERT(t, "alloc");
    int64_t big[1] = {7};   /* contract for dim1 is {2} */
    ce_tensor_t* out = ce_tensor_zeros(big, 1);
    ASSERT(out, "alloc");

    ce_status_t st = ce_get_default_backend()->sum(t, out, 1);
    if (st == CE_OK) {
        FAIL("sum accepted a wrong-sized output tensor");
        return;
    }
    ASSERT(st == CE_ERR_INVALID_SHAPE, "sum must reject bad output size");

    ce_tensor_release(t); ce_tensor_release(out);
    PASS();
}

/* ---- RED (heap corruption on old code): 3D dim2 sum/mean ---- */
static void test_3d_sum_and_mean_dim2(void) {
    TEST("sum/mean: 3D dim2 reduces axis2 (was OOB)");
    float d[24];
    fill_seq(d, 24);
    ce_tensor_t* t = make3d(d, 2, 3, 4);
    ASSERT(t, "alloc");

    int64_t sh[2] = {2, 3};
    ce_tensor_t* out = ce_tensor_zeros(sh, 2);
    ASSERT(out, "alloc");
    ASSERT(ce_get_default_backend()->sum(t, out, 2) == CE_OK, "dim2 sum");
    float ref[6];
    ref_reduce(t, 2, false, ref);
    float md = max_diff(out->data, ref, 6);
    if (md > 1e-4f) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "sum max|diff|=%g — dim2 must reduce axis2", (double)md);
        FAIL(buf);
        return;
    }

    ce_tensor_t* mout = ce_tensor_zeros(sh, 2);
    ASSERT(mout, "alloc");
    ASSERT(ce_get_default_backend()->mean(t, mout, 2) == CE_OK, "dim2 mean");
    float mref[6];
    for (int i = 0; i < 6; i++) mref[i] = ref[i] / 4.0f;
    md = max_diff(mout->data, mref, 6);
    if (md > 1e-4f) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "mean max|diff|=%g — dim2 must reduce axis2", (double)md);
        FAIL(buf);
        return;
    }

    ce_tensor_release(t); ce_tensor_release(out); ce_tensor_release(mout);
    PASS();
}

int main(void) {
    /* Unbuffered: these tests include heap-corruption RED cases —
     * keep the already-run results visible if a later one aborts. */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\n=== CEngine Reduction Dim Tests ===\n\n");
    ce_init();

    test_legacy_2d_and_3d_dim0();
    test_3d_sum_dim1();
    test_3d_max_dims();
    test_invalid_dims();
    test_output_shape_guard();
    test_3d_sum_and_mean_dim2();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
