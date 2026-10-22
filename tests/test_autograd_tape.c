/*
 * test_autograd_tape.c — tape restriction tests (major D4).
 *
 * D4: ce_tape_backward only ever processed the LAST entry — earlier
 * entries keep grad == NULL from record() and were silently skipped
 * (`if (!entry->backward_fn) continue` after the NULL-grad check), and
 * layer tape-backwards don't compute grad_input at all (linear's
 * accumulates parameter grads only).  The chain therefore yielded
 * partial gradients with a success status.  Nothing in-tree calls the
 * tape backward (trainer and Python both use model->backward), so the
 * approved fix is to REFUSE multi-entry chains loudly
 * (CE_ERR_NOT_IMPLEMENTED) instead of pretending; single-entry and
 * empty tapes stay well-defined.
 */
#include "cengine.h"
#include "autograd.h"
#include "layers.h"
#include "tensor.h"
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

static void noop_backward(void* state, ce_tensor_t* grad) {
    (void)state; (void)grad;
}

/* ---- D4: multi-entry chains refuse; ≤1 entries stay defined ---- */

static void test_multientry_chain_is_refused(void) {
    TEST("tape: multi-entry chain refused (not partial grads)");

    ce_tape_t* tape = ce_tape_create();
    ASSERT(tape != NULL, "tape create");
    ASSERT(ce_tape_start(tape) == CE_OK, "tape start");

    int64_t sh[2] = {1, 4};
    ce_tensor_t* o1 = ce_tensor_zeros(sh, 2);
    ce_tensor_t* o2 = ce_tensor_zeros(sh, 2);
    ASSERT(o1 && o2, "tensor alloc");

    ASSERT(ce_tape_record(tape, o1, noop_backward, NULL, 1) == CE_OK,
           "record 1");
    ASSERT(ce_tape_record(tape, o2, noop_backward, NULL, 1) == CE_OK,
           "record 2");
    ASSERT(ce_tape_stop(tape) == CE_OK, "tape stop");

    float g[4] = {1, 1, 1, 1};
    ce_tensor_t* loss_grad = ce_tensor_create_from_data(g, sh, 2);
    ASSERT(loss_grad, "loss_grad alloc");

    ce_status_t st = ce_tape_backward(tape, loss_grad);
    ASSERT(st == CE_ERR_NOT_IMPLEMENTED,
           "multi-entry backward must refuse — it used to process only "
           "the last entry and return CE_OK with partial gradients");

    ce_tensor_release(loss_grad);
    ce_tensor_release(o1);
    ce_tensor_release(o2);
    ce_tape_destroy(tape);
    PASS();
}

static void test_single_and_empty_entries_remain_defined(void) {
    TEST("tape: single-entry and empty tapes still run");

    ce_tape_t* tape = ce_tape_create();
    ASSERT(tape != NULL, "tape create");
    ASSERT(ce_tape_start(tape) == CE_OK, "tape start");
    ASSERT(ce_tape_backward(tape, NULL) == CE_OK, "empty tape backward");

    int64_t sh[2] = {1, 4};
    ce_tensor_t* o1 = ce_tensor_zeros(sh, 2);
    ASSERT(o1, "tensor alloc");
    ASSERT(ce_tape_record(tape, o1, noop_backward, NULL, 1) == CE_OK,
           "record 1");

    float g[4] = {1, 2, 3, 4};
    ce_tensor_t* loss_grad = ce_tensor_create_from_data(g, sh, 2);
    ASSERT(loss_grad, "loss_grad alloc");
    ASSERT(ce_tape_backward(tape, loss_grad) == CE_OK,
           "single-entry backward must remain supported");

    ce_tensor_release(loss_grad);
    ce_tensor_release(o1);
    ce_tape_destroy(tape);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Autograd Tape Tests ===\n\n");
    ce_init();

    test_multientry_chain_is_refused();
    test_single_and_empty_entries_remain_defined();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
