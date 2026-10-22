/*
 * test_autograd_tape.c — tape restriction (major D4) and state-leak
 * regression tests (major D5).
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
 *
 * D5: entries hold layer states that RETAIN tensors (linear keeps
 * input_cache/weight/bias/grad_w/grad_b, activations keep their
 * output cache), but ce_tape_start/destroy released only entry->output
 * and entry->grad, then free(state) — the retained references leaked
 * forever.  Each record now carries a state destructor.
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

    ASSERT(ce_tape_record(tape, o1, noop_backward, NULL, 1, NULL) == CE_OK,
           "record 1");
    ASSERT(ce_tape_record(tape, o2, noop_backward, NULL, 1, NULL) == CE_OK,
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
    ASSERT(ce_tape_record(tape, o1, noop_backward, NULL, 1, NULL) == CE_OK,
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

/* ---- D5: recorded states must release the tensors they retain ---- */

static void test_recorded_state_releases_retained_tensors(void) {
    TEST("tape: start releases state-retained tensors");

    ce_tape_t* tape = ce_tape_create();
    ASSERT(tape != NULL, "tape create");
    ce_set_tape(tape);
    ASSERT(ce_tape_start(tape) == CE_OK, "tape start");

    /* Linear forward under recording: its state retains input_cache,
     * weight, bias, grad_w, grad_b. */
    ce_module_t* lin = ce_linear_create(4, 3, "tape_lin");
    ASSERT(lin != NULL, "linear create");
    ce_linear_t* st = (ce_linear_t*)lin->internal_state;
    ASSERT(st != NULL && st->input_cache == NULL, "precondition");

    int64_t in_sh[2] = {2, 4};
    int64_t out_sh[2] = {2, 3};
    float in_data[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    ce_tensor_t* input  = ce_tensor_create_from_data(in_data, in_sh, 2);
    ce_tensor_t* output = ce_tensor_zeros(out_sh, 2);
    ASSERT(input && output, "tensor alloc");

    ASSERT(ce_linear_forward(lin, input, output) == CE_OK, "forward");
    ASSERT(st->input_cache != NULL, "input_cache set");
    /* clone in forward (+1) + retain in tape state (+1) = 2 */
    int64_t rc_after_record = st->input_cache->refcount;
    ASSERT(rc_after_record == 2,
           "precondition: state retains input_cache (refcount 2)");

    /* Clearing the tape must release the state's reference. */
    ASSERT(ce_tape_start(tape) == CE_OK, "tape start (clear)");
    if (st->input_cache->refcount != 1) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "input_cache refcount = %lld after tape clear "
                 "(state retained tensor leaked)",
                 (long long)st->input_cache->refcount);
        FAIL(buf);
        ce_tensor_release(input);
        ce_tensor_release(output);
        ce_module_destroy(lin);
        ce_set_tape(NULL);
        ce_tape_destroy(tape);
        return;
    }

    ce_tensor_release(input);
    ce_tensor_release(output);
    ce_module_destroy(lin);
    ce_set_tape(NULL);
    ce_tape_destroy(tape);
    PASS();
}

static void test_destroy_releases_state_tensors(void) {
    TEST("tape: destroy releases state-retained tensors");

    ce_tape_t* tape = ce_tape_create();
    ASSERT(tape != NULL, "tape create");
    ce_set_tape(tape);
    ASSERT(ce_tape_start(tape) == CE_OK, "tape start");

    ce_module_t* lin = ce_linear_create(4, 3, "tape_lin2");
    ASSERT(lin != NULL, "linear create");
    ce_linear_t* st = (ce_linear_t*)lin->internal_state;

    int64_t in_sh[2] = {2, 4};
    int64_t out_sh[2] = {2, 3};
    float in_data[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    ce_tensor_t* input  = ce_tensor_create_from_data(in_data, in_sh, 2);
    ce_tensor_t* output = ce_tensor_zeros(out_sh, 2);
    ASSERT(input && output, "tensor alloc");

    ASSERT(ce_linear_forward(lin, input, output) == CE_OK, "forward");
    ASSERT(st->input_cache != NULL, "input_cache set");
    ASSERT(st->input_cache->refcount == 2, "precondition (retained)");

    ce_set_tape(NULL);
    ce_tape_destroy(tape);   /* must run the state destructor */

    int64_t rc = st->input_cache->refcount;
    if (rc != 1) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "input_cache refcount = %lld after tape_destroy "
                 "(state retained tensor leaked)", (long long)rc);
        FAIL(buf);
        ce_tensor_release(input);
        ce_tensor_release(output);
        ce_module_destroy(lin);
        return;
    }

    ce_tensor_release(input);
    ce_tensor_release(output);
    ce_module_destroy(lin);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Autograd Tape Tests ===\n\n");
    ce_init();

    test_multientry_chain_is_refused();
    test_single_and_empty_entries_remain_defined();
    test_recorded_state_releases_retained_tensors();
    test_destroy_releases_state_tensors();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
