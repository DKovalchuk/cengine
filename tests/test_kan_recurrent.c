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

int main(void) {
    printf("\n=== CEngine Recurrent-KAN Tests ===\n\n");
    ce_init();

    test_parallel_matches_serial();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
