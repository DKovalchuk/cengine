/*
 * test_softmax_backward.c — softmax backward (major C3).
 *
 * ce_softmax_backward() used to be a no-op (`(void)grad_input; return
 * CE_OK;`) — in the production (non-tape) path the layer never wrote
 * grad_input, so every model with a softmax in the chain received stale
 * or zero gradients.  The tape path had the correct Jacobian-vector
 * product all along; the module path now delegates to the same backend
 * implementation the tape formula matches:
 *
 *     grad_in_i = y_i * (g_i - sum_j g_j * y_j)     (per row)
 *
 * State mirrors the sigmoid layer: forward caches its output, backward
 * uses it; backward without a prior forward fails with
 * CE_ERR_NOT_INITIALIZED instead of pretending success.
 */
#include "cengine.h"
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

#define SENTINEL 7.77f
#define TOL 1e-6f

/* Row-wise JVP reference computed from the ACTUAL forward output. */
static float jvp_reference(const ce_tensor_t* y, const ce_tensor_t* g,
                           float* out) {
    int64_t batch   = (y->ndim >= 2) ? y->shape[0] : 1;
    int64_t classes = y->numel / batch;
    float maxd = 0.0f;
    for (int64_t b = 0; b < batch; b++) {
        const float* yrow = &y->data[b * classes];
        const float* grow = &g->data[b * classes];
        float dot = 0.0f;
        for (int64_t c = 0; c < classes; c++) dot += grow[c] * yrow[c];
        for (int64_t c = 0; c < classes; c++)
            out[b * classes + c] = yrow[c] * (grow[c] - dot);
    }
    return maxd;
}

static void fill_pattern(float* p, int64_t n, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
    }
}

static void test_softmax_backward_jvp(void) {
    TEST("softmax backward: JVP on saved output (sentinel must go)");
    const int64_t B = 2, C = 4;

    ce_module_t* mod = ce_softmax_create("sm_bw");
    ASSERT(mod != NULL, "create failed");

    int64_t sh[2] = {B, C};
    float in_data[8], g_data[8];
    fill_pattern(in_data, 8, 111u);
    fill_pattern(g_data, 8, 222u);
    ce_tensor_t* in    = ce_tensor_create_from_data(in_data, sh, 2);
    ce_tensor_t* out   = ce_tensor_zeros(sh, 2);
    ce_tensor_t* g_out = ce_tensor_create_from_data(g_data, sh, 2);
    ce_tensor_t* g_in  = ce_tensor_zeros(sh, 2);
    ASSERT(in && out && g_out && g_in, "tensor alloc");

    /* Poison grad_input: the old no-op backward left this untouched. */
    for (int64_t i = 0; i < g_in->numel; i++) g_in->data[i] = SENTINEL;

    ASSERT(ce_softmax_forward(mod, in, out) == CE_OK, "forward failed");
    ASSERT(ce_softmax_backward(mod, g_out, g_in) == CE_OK, "backward failed");

    float ref[8];
    jvp_reference(out, g_out, ref);

    float maxd = 0.0f;
    for (int64_t i = 0; i < 8; i++) {
        float d = fabsf(g_in->data[i] - ref[i]);
        if (d > maxd) maxd = d;
    }
    if (maxd > TOL) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "max|diff|=%g — grad_input not written or wrong "
                 "(no-op backward leaves %g), want <%g",
                 (double)maxd, (double)SENTINEL, (double)TOL);
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

static void test_backward_without_forward_fails(void) {
    TEST("softmax backward before forward → NOT_INITIALIZED");

    ce_module_t* mod = ce_softmax_create("sm_nofwd");
    ASSERT(mod != NULL, "create failed");

    int64_t sh[2] = {1, 4};
    float zeros[4] = {0, 0, 0, 0};
    ce_tensor_t* g_out = ce_tensor_create_from_data(zeros, sh, 2);
    ce_tensor_t* g_in  = ce_tensor_zeros(sh, 2);
    ASSERT(g_out && g_in, "tensor alloc");

    ce_status_t st = ce_softmax_backward(mod, g_out, g_in);
    ASSERT(st == CE_ERR_NOT_INITIALIZED,
           "backward without a forward must fail (used to return CE_OK "
           "while writing nothing)");

    ASSERT(ce_softmax_backward(NULL, g_out, g_in) == CE_ERR_INVALID_ARG,
           "NULL module guard");

    ce_tensor_release(g_out);
    ce_tensor_release(g_in);
    ce_module_destroy(mod);
    PASS();
}

static void test_gumbel_backward_jvp(void) {
    TEST("gumbel-softmax backward: same JVP via cached output");

    ce_module_t* mod = ce_gumbel_softmax_create(1.0f, "gb_bw");
    ASSERT(mod != NULL, "create failed");

    int64_t sh[2] = {2, 4};
    float in_data[8], g_data[8];
    fill_pattern(in_data, 8, 333u);
    fill_pattern(g_data, 8, 444u);
    ce_tensor_t* in    = ce_tensor_create_from_data(in_data, sh, 2);
    ce_tensor_t* out   = ce_tensor_zeros(sh, 2);
    ce_tensor_t* g_out = ce_tensor_create_from_data(g_data, sh, 2);
    ce_tensor_t* g_in  = ce_tensor_zeros(sh, 2);
    ASSERT(in && out && g_out && g_in, "tensor alloc");
    for (int64_t i = 0; i < g_in->numel; i++) g_in->data[i] = SENTINEL;

    ASSERT(ce_softmax_forward(mod, in, out) == CE_OK, "forward failed");
    ASSERT(ce_softmax_backward(mod, g_out, g_in) == CE_OK, "backward failed");

    float ref[8];
    jvp_reference(out, g_out, ref);
    float maxd = 0.0f;
    for (int64_t i = 0; i < 8; i++) {
        float d = fabsf(g_in->data[i] - ref[i]);
        if (d > maxd) maxd = d;
    }
    ASSERT(maxd <= TOL, "gumbel grad_input not written or wrong");

    ce_tensor_release(in);
    ce_tensor_release(out);
    ce_tensor_release(g_out);
    ce_tensor_release(g_in);
    ce_module_destroy(mod);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Softmax Backward Tests ===\n\n");
    ce_init();

    test_softmax_backward_jvp();
    test_backward_without_forward_fails();
    test_gumbel_backward_jvp();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
