/*
 * test_dataloader.c — dataloader generator-reconfiguration tests.
 *
 * Regression for a data race (major A1): the training loop mutated the
 * shared generator (rng_state / n_ops / ops) directly while the dataloader's
 * prefetch worker thread was reading it in ce_alu_gen_batch().  The fix is
 * ce_dataloader_reseed(): it joins the worker, applies the new seed and op
 * filter, discards batches generated under the previous settings and
 * restarts prefetch — so the batch returned by the next
 * ce_dataloader_next() is generated with the new settings.
 */
#include "cengine.h"
#include "data.h"
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

static int sample_equal(const ce_alu_sample_t* a, const ce_alu_sample_t* b) {
    if (a->n_bits != b->n_bits || a->op != b->op) return 0;
    const ce_tensor_t* ta[3] = { a->a_bits, a->b_bits, a->result_bits };
    const ce_tensor_t* tb[3] = { b->a_bits, b->b_bits, b->result_bits };
    for (int k = 0; k < 3; k++) {
        if (ta[k]->numel != tb[k]->numel) return 0;
        for (int64_t i = 0; i < ta[k]->numel; i++) {
            if (ta[k]->data[i] != tb[k]->data[i]) return 0;
        }
    }
    return 1;
}

static int batch_equal(const ce_batch_t* a, const ce_batch_t* b) {
    if (!a || !b || a->count != b->count) return 0;
    for (int i = 0; i < a->count; i++) {
        if (!sample_equal(a->samples[i], b->samples[i])) return 0;
    }
    return 1;
}

/* After reseed(seed), the next batch must equal the batch produced by a
 * freshly created generator with the same seed (bit-for-bit). */
static void test_reseed_matches_direct_generation(void) {
    TEST("reseed: next batch == fresh generator with same seed");
    const int ops[4] = { CE_OP_ADD, CE_OP_SUB, CE_OP_AND, CE_OP_XOR };

    ce_alu_gen_t* gen = ce_alu_gen_create(4, 8, ops, 4, 7);
    ASSERT(gen != NULL, "gen create");
    ce_dataloader_t* dl = ce_dataloader_create(gen, 4, 16, 7);
    ASSERT(dl != NULL, "dataloader create");

    /* Reference: batch drawn directly from a fresh generator seeded 42. */
    ce_alu_gen_t* ref_gen = ce_alu_gen_create(4, 8, ops, 4, 42);
    ASSERT(ref_gen != NULL, "ref gen create");
    ce_batch_t* ref = ce_alu_gen_batch(ref_gen, 4);
    ASSERT(ref != NULL, "ref batch");

    ce_status_t st = ce_dataloader_reseed(dl, 42, NULL, 0);
    ASSERT(st == CE_OK, "reseed failed");

    ce_batch_t* got = ce_dataloader_next(dl);
    ASSERT(got != NULL, "dataloader next after reseed");
    ASSERT(batch_equal(got, ref),
           "batch after reseed differs from fresh generator with same seed");

    ce_batch_destroy(got);
    ce_batch_destroy(ref);
    free(ref_gen);
    ce_dataloader_destroy(dl);
    free(gen);
    PASS();
}

/* An op filter passed to reseed must apply to the very next batch. */
static void test_reseed_applies_ops_filter(void) {
    TEST("reseed: ops filter applies to the next batch");
    const int ops[4] = { CE_OP_ADD, CE_OP_SUB, CE_OP_AND, CE_OP_XOR };

    ce_alu_gen_t* gen = ce_alu_gen_create(4, 8, ops, 4, 7);
    ASSERT(gen != NULL, "gen create");
    ce_dataloader_t* dl = ce_dataloader_create(gen, 4, 16, 7);
    ASSERT(dl != NULL, "dataloader create");

    const int and_only[1] = { CE_OP_AND };
    ce_status_t st = ce_dataloader_reseed(dl, 42, and_only, 1);
    ASSERT(st == CE_OK, "reseed with ops failed");

    ce_batch_t* got = ce_dataloader_next(dl);
    ASSERT(got != NULL, "dataloader next");
    ASSERT(got->count > 0, "empty batch");
    for (int i = 0; i < got->count; i++) {
        ASSERT(got->samples[i]->op == CE_OP_AND,
               "sample op not from the reseeded filter");
    }

    ce_batch_destroy(got);
    ce_dataloader_destroy(dl);
    free(gen);
    PASS();
}

/* Boundary/invalid arguments must be rejected without side effects. */
static void test_reseed_invalid_args(void) {
    TEST("reseed: invalid arguments return CE_ERR_INVALID_ARG");
    const int ops[4] = { CE_OP_ADD, CE_OP_SUB, CE_OP_AND, CE_OP_XOR };

    ce_alu_gen_t* gen = ce_alu_gen_create(4, 8, ops, 4, 7);
    ASSERT(gen != NULL, "gen create");
    ce_dataloader_t* dl = ce_dataloader_create(gen, 4, 16, 7);
    ASSERT(dl != NULL, "dataloader create");

    ASSERT(ce_dataloader_reseed(NULL, 42, NULL, 0) == CE_ERR_INVALID_ARG,
           "NULL dataloader not rejected");
    ASSERT(ce_dataloader_reseed(dl, 42, ops, 0) == CE_ERR_INVALID_ARG,
           "ops!=NULL with n_ops=0 not rejected");
    ASSERT(ce_dataloader_reseed(dl, 42, ops, CE_NUM_OPS + 1) == CE_ERR_INVALID_ARG,
           "n_ops beyond CE_NUM_OPS not rejected");
    const int bad_op[1] = { CE_NUM_OPS };
    ASSERT(ce_dataloader_reseed(dl, 42, bad_op, 1) == CE_ERR_INVALID_ARG,
           "out-of-range op not rejected");

    ce_dataloader_destroy(dl);
    free(gen);
    PASS();
}

/* Stress: reseed must safely join a worker that is running (or has just
 * finished) every time; also covers the concurrent-window regression. */
static void test_reseed_stress(void) {
    TEST("reseed: 500 reseed/next cycles join cleanly");
    const int ops[4] = { CE_OP_ADD, CE_OP_SUB, CE_OP_AND, CE_OP_XOR };

    ce_alu_gen_t* gen = ce_alu_gen_create(4, 8, ops, 4, 7);
    ASSERT(gen != NULL, "gen create");
    ce_dataloader_t* dl = ce_dataloader_create(gen, 8, 16, 7);
    ASSERT(dl != NULL, "dataloader create");

    for (int i = 0; i < 500; i++) {
        int cycle_ops[2] = { CE_OP_ADD, i % 2 ? CE_OP_XOR : CE_OP_AND };
        ce_status_t st = ce_dataloader_reseed(
            dl, 1000 + i, (i % 3 == 0) ? cycle_ops : NULL,
            (i % 3 == 0) ? 2 : 0);
        ASSERT(st == CE_OK, "reseed failed in stress loop");
        ce_batch_t* b = ce_dataloader_next(dl);
        ASSERT(b != NULL, "dataloader next failed in stress loop");
        ce_batch_destroy(b);
    }

    ce_dataloader_destroy(dl);
    free(gen);
    PASS();
}

int main(void) {
    printf("\n=== CEngine DataLoader Tests ===\n\n");
    ce_init();

    test_reseed_matches_direct_generation();
    test_reseed_applies_ops_filter();
    test_reseed_invalid_args();
    test_reseed_stress();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
