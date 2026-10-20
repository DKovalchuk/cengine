#include "cengine.h"
#include "memory.h"
#include "tensor.h"
#include "compute.h"
#include "simd.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  TEST %-40s ", name); \
    tests_run++; \
} while(0)

#define PASS() do { tests_passed++; printf("[PASS]\n"); } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); } while(0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "%s: got %f, expected %f", msg, (double)(a), (double)(b)); \
        FAIL(buf); return; \
    } \
} while(0)

/* ---- Test: init/shutdown ---- */
static void test_init_shutdown(void) {
    TEST("init/shutdown");
    ce_status_t st = ce_init();
    ASSERT(st == CE_OK, "ce_init failed");
    ce_shutdown();
    /* Re-init is OK */
    st = ce_init();
    ASSERT(st == CE_OK, "ce_init re-init failed");
    PASS();
}

/* ---- Test: memory tracker ---- */
static void test_memory_tracker(void) {
    TEST("memory tracker");
    ce_tracker_reset_rates();

    void* p1 = ce_aligned_alloc(64, 1024);
    ASSERT(p1 != NULL, "alloc failed");
    ASSERT(((uintptr_t)p1 % 64) == 0, "not aligned");

    ce_memory_stats_t stats = ce_tracker_get_stats();
    ASSERT(stats.current_bytes >= 1024, "current_bytes wrong");
    ASSERT(stats.total_allocs >= 1, "total_allocs wrong");
    ASSERT(stats.active_allocs >= 1, "active_allocs wrong");

    void* p2 = ce_aligned_alloc(64, 2048);
    stats = ce_tracker_get_stats();
    ASSERT(stats.active_allocs >= 2, "active_allocs after 2nd alloc");
    ASSERT(stats.peak_bytes >= stats.current_bytes, "peak < current");

    ce_aligned_free(p1);
    stats = ce_tracker_get_stats();
    ASSERT(stats.active_allocs >= 1, "active_allocs after free");

    ce_aligned_free(p2);
    stats = ce_tracker_get_stats();
    ASSERT(stats.active_allocs == 0, "active_allocs after all freed");
    PASS();
}

/* ---- Test: memory pool ---- */
static void test_memory_pool(void) {
    TEST("memory pool");
    ce_mem_pool_t* pool = ce_mem_pool_create(65536);
    ASSERT(pool != NULL, "pool create failed");

    void* p1 = ce_mem_pool_alloc(pool, 64);
    ASSERT(p1 != NULL, "pool alloc 64 failed");

    void* p2 = ce_mem_pool_alloc(pool, 256);
    ASSERT(p2 != NULL, "pool alloc 256 failed");

    ce_mem_pool_free(pool, p1);
    /* Reuse from free list */
    void* p3 = ce_mem_pool_alloc(pool, 64);
    ASSERT(p3 == p1, "pool did not reuse freed block");

    ce_mem_pool_reset(pool);
    void* p4 = ce_mem_pool_alloc(pool, 65536);
    ASSERT(p4 != NULL, "pool alloc after reset failed");

    ce_mem_pool_destroy(pool);
    PASS();
}

/* ---- Test: tensor create/destroy ---- */
static void test_tensor_lifecycle(void) {
    TEST("tensor create/destroy");
    int64_t shape[] = {4, 8};
    ce_tensor_t* t = ce_tensor_create(shape, 2, CE_FLOAT32);
    ASSERT(t != NULL, "create failed");
    ASSERT(t->ndim == 2, "ndim wrong");
    ASSERT(t->shape[0] == 4, "shape[0] wrong");
    ASSERT(t->shape[1] == 8, "shape[1] wrong");
    ASSERT(t->numel == 32, "numel wrong");
    ASSERT(t->data != NULL, "data is NULL");
    ASSERT(t->refcount == 1, "refcount wrong");

    ce_tensor_retain(t);
    ASSERT(t->refcount == 2, "retain failed");
    ce_tensor_release(t);
    ASSERT(t->refcount == 1, "release failed");
    ce_tensor_release(t); /* frees */
    PASS();
}

/* ---- Test: tensor operations ---- */
static void test_tensor_ops(void) {
    TEST("tensor ops (fill, get, set)");
    int64_t shape[] = {3, 4};
    ce_tensor_t* t = ce_tensor_zeros(shape, 2);

    ce_tensor_fill(t, 3.14f);
    int64_t idx[] = {1, 2};
    ASSERT_NEAR(ce_tensor_get(t, idx), 3.14f, 1e-5f, "fill/get");

    ce_tensor_set(t, idx, 42.0f);
    ASSERT_NEAR(ce_tensor_get(t, idx), 42.0f, 1e-5f, "set/get");

    ce_tensor_t* c = ce_tensor_clone(t);
    ASSERT(c != NULL, "clone failed");
    ASSERT_NEAR(ce_tensor_get(c, idx), 42.0f, 1e-5f, "clone data");

    ce_tensor_release(t);
    ce_tensor_release(c);
    PASS();
}

/* ---- Test: matmul ---- */
static void test_matmul(void) {
    TEST("matmul (AVX2)");
    /* A[2,3] = [[1,2,3],[4,5,6]] */
    /* B[3,2] = [[7,8],[9,10],[11,12]] */
    /* Expected C = A*B = [[58,64],[139,154]] */
    float a_data[] = {1,2,3, 4,5,6};
    float b_data[] = {7,8, 9,10, 11,12};

    int64_t a_shape[] = {2, 3};
    int64_t b_shape[] = {3, 2};
    int64_t c_shape[] = {2, 2};

    ce_tensor_t* A = ce_tensor_create_from_data(a_data, a_shape, 2);
    ce_tensor_t* B = ce_tensor_create_from_data(b_data, b_shape, 2);
    ce_tensor_t* C = ce_tensor_zeros(c_shape, 2);

    ce_status_t st = ce_matmul(A, B, C, CE_TRANSPOSE_NONE);
    ASSERT(st == CE_OK, "matmul failed");

    int64_t i00[] = {0, 0}, i01[] = {0, 1};
    int64_t i10[] = {1, 0}, i11[] = {1, 1};

    ASSERT_NEAR(ce_tensor_get(C, i00), 58.0f, 1e-4f, "C[0,0]");
    ASSERT_NEAR(ce_tensor_get(C, i01), 64.0f, 1e-4f, "C[0,1]");
    ASSERT_NEAR(ce_tensor_get(C, i10), 139.0f, 1e-4f, "C[1,0]");
    ASSERT_NEAR(ce_tensor_get(C, i11), 154.0f, 1e-4f, "C[1,1]");

    ce_tensor_release(A);
    ce_tensor_release(B);
    ce_tensor_release(C);
    PASS();
}

/* ---- Test: activations ---- */
static void test_activations(void) {
    TEST("activations (sigmoid, tanh, elu, relu)");
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    int64_t shape[] = {1, 5};
    ce_tensor_t* in = ce_tensor_create_from_data(data, shape, 2);
    ce_tensor_t* out = ce_tensor_zeros(shape, 2);

    /* Sigmoid */
    ce_act(in, CE_ACT_SIGMOID, 0, 0, out);
    ASSERT_NEAR(ce_tensor_get(out, (int64_t[]){0,0}), ce_sigmoid_scalar(-2.0f), 1e-4f, "sigmoid(-2)");
    ASSERT_NEAR(ce_tensor_get(out, (int64_t[]){0,2}), 0.5f, 1e-4f, "sigmoid(0)");
    ASSERT_NEAR(ce_tensor_get(out, (int64_t[]){0,4}), ce_sigmoid_scalar(2.0f), 1e-4f, "sigmoid(2)");

    /* ELU */
    ce_act(in, CE_ACT_ELU, 1.0f, 0, out);
    ASSERT_NEAR(ce_tensor_get(out, (int64_t[]){0,2}), 0.0f, 1e-4f, "elu(0)");
    ASSERT_NEAR(ce_tensor_get(out, (int64_t[]){0,4}), 2.0f, 1e-4f, "elu(2)");

    /* Tanh */
    ce_act(in, CE_ACT_TANH, 0, 0, out);
    ASSERT_NEAR(ce_tensor_get(out, (int64_t[]){0,2}), 0.0f, 1e-4f, "tanh(0)");
    ASSERT_NEAR(ce_tensor_get(out, (int64_t[]){0,4}), tanhf(2.0f), 1e-3f, "tanh(2)");

    /* ReLU */
    ce_act(in, CE_ACT_RELU, 0, 0, out);
    ASSERT_NEAR(ce_tensor_get(out, (int64_t[]){0,0}), 0.0f, 1e-4f, "relu(-2)");
    ASSERT_NEAR(ce_tensor_get(out, (int64_t[]){0,4}), 2.0f, 1e-4f, "relu(2)");

    ce_tensor_release(in);
    ce_tensor_release(out);
    PASS();
}

/* ---- Test: reduction ---- */
static void test_reductions(void) {
    TEST("reductions (sum, norm, dot)");
    float data[] = {1, 2, 3, 4};
    int64_t shape[] = {2, 2};
    ce_tensor_t* t = ce_tensor_create_from_data(data, shape, 2);

    /* Norm */
    float n = ce_get_default_backend()->norm(t);
    ASSERT_NEAR(n, sqrtf(30.0f), 1e-4f, "norm"); /* sqrt(1+4+9+16) */

    /* Dot */
    ce_tensor_t* t2 = ce_tensor_create_from_data(data, shape, 2);
    float d = ce_get_default_backend()->dot(t, t2);
    ASSERT_NEAR(d, 30.0f, 1e-4f, "dot"); /* 1+4+9+16 */

    ce_tensor_release(t);
    ce_tensor_release(t2);
    PASS();
}

/* ---- Test: softmax ---- */
static void test_softmax(void) {
    TEST("softmax");
    float data[] = {1.0f, 2.0f, 3.0f};
    int64_t shape[] = {1, 3};
    ce_tensor_t* in = ce_tensor_create_from_data(data, shape, 2);
    ce_tensor_t* out = ce_tensor_zeros(shape, 2);

    ce_act(in, CE_ACT_SOFTMAX, 0, 0, out);

    /* Sum should be 1.0 */
    float sum = 0;
    for (int i = 0; i < 3; i++) {
        int64_t idx[] = {0, i};
        sum += ce_tensor_get(out, idx);
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5f, "softmax sum");

    /* Monotonic: out[2] > out[1] > out[0] */
    ASSERT(ce_tensor_get(out, (int64_t[]){0,2}) > ce_tensor_get(out, (int64_t[]){0,1}),
           "softmax monotonic 2>1");
    ASSERT(ce_tensor_get(out, (int64_t[]){0,1}) > ce_tensor_get(out, (int64_t[]){0,0}),
           "softmax monotonic 1>0");

    ce_tensor_release(in);
    ce_tensor_release(out);
    PASS();
}

/* ---- Test: gradient backward ---- */
static void test_backward_sigmoid(void) {
    TEST("backward sigmoid");
    /* act_out = sigmoid(x) */
    float act_data[] = {0.1f, 0.5f, 0.9f};
    float grad_data[] = {1.0f, 1.0f, 1.0f};
    int64_t shape[] = {1, 3};

    ce_tensor_t* act_out = ce_tensor_create_from_data(act_data, shape, 2);
    ce_tensor_t* grad_out = ce_tensor_create_from_data(grad_data, shape, 2);
    ce_tensor_t* grad_in = ce_tensor_zeros(shape, 2);

    ce_act_bw(grad_out, act_out, CE_ACT_SIGMOID, 0, grad_in);

    /* d/dx sigmoid = y * (1-y) */
    ASSERT_NEAR(ce_tensor_get(grad_in, (int64_t[]){0,0}), 0.1f * 0.9f, 1e-5f, "sig bw 0");
    ASSERT_NEAR(ce_tensor_get(grad_in, (int64_t[]){0,1}), 0.5f * 0.5f, 1e-5f, "sig bw 1");
    ASSERT_NEAR(ce_tensor_get(grad_in, (int64_t[]){0,2}), 0.9f * 0.1f, 1e-5f, "sig bw 2");

    ce_tensor_release(act_out);
    ce_tensor_release(grad_out);
    ce_tensor_release(grad_in);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Phase 1 Tests ===\n\n");

    ce_init();

    test_init_shutdown();
    ce_init(); /* re-init after shutdown */

    test_memory_tracker();
    test_memory_pool();
    test_tensor_lifecycle();
    test_tensor_ops();
    test_matmul();
    test_activations();
    test_reductions();
    test_softmax();
    test_backward_sigmoid();

    ce_shutdown();

    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
