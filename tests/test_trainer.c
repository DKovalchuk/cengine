/*
 * test_trainer.c — trainer input-layout regression tests.
 *
 * Regression for a heap OOB write in prepare_batch() (trainer.c):
 * op_code is a 3-float vector ([3], 3-bit encoding incl. CE_OP_OR=4),
 * but input_dim was 2*max_bits+2, so the memcpy of 3 floats wrote the
 * third float one past the end of every row — past the input buffer for
 * the last row — and the model never saw op bit 2.
 *
 * The probe module below records the input row width the trainer
 * actually hands to model->forward.
 */
#include "cengine.h"
#include "memory.h"
#include "tensor.h"
#include "layers.h"
#include "trainer.h"
#include "../src/training/trainer_internal.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

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

/* ---- Probe module: records the input row width it receives ---- */

static int g_seen_width = -1;
static int g_forward_calls = 0;

static ce_status_t probe_forward(struct ce_module* self,
                                 ce_tensor_t* input, ce_tensor_t* output) {
    (void)self;
    g_forward_calls++;
    g_seen_width = (input && input->ndim == 2) ? (int)input->shape[1] : -1;
    ce_tensor_zero(output);
    return CE_OK;
}

static ce_status_t probe_backward(struct ce_module* self,
                                  ce_tensor_t* grad_output,
                                  ce_tensor_t* grad_input) {
    (void)self; (void)grad_output; (void)grad_input;
    return CE_OK;
}

static ce_module_t* make_probe_module(void) {
    ce_module_t* mod = ce_module_create("probe");
    if (!mod) return NULL;

    int64_t pshape[1] = {1};
    ce_tensor_t* p = ce_tensor_zeros(pshape, 1);
    ce_tensor_t* g = ce_tensor_zeros(pshape, 1);
    if (!p || !g) return NULL;
    if (ce_module_add_parameter(mod, p, g) != CE_OK) return NULL;

    mod->forward = probe_forward;
    mod->backward = probe_backward;
    mod->zero_grad = ce_module_default_zero_grad;
    return mod;
}

/* ---- Test: op_code(3) must fit in the input row ---- */
static void test_trainer_op_code_layout(void) {
    TEST("trainer input row fits 3 op bits (2*mb+3)");
    const int max_bits = 8;

    ce_module_t* probe = make_probe_module();
    ASSERT(probe != NULL, "probe module create");

    ce_trainer_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.model              = probe;
    cfg.loss_type          = CE_LOSS_BCE;
    cfg.optimizer_type     = CE_OPT_ADAM;
    cfg.learning_rate      = 1e-3f;
    cfg.total_epochs       = 1;
    cfg.batch_size         = 4;
    cfg.samples_per_epoch  = 8;
    cfg.seed               = 42;
    cfg.max_bits           = max_bits;
    cfg.loss_threshold     = 0.0f;  /* never advance curriculum here */
    cfg.patience           = 1;
    cfg.max_epochs_per_stage = 100;

    ce_trainer_t* t = ce_trainer_create(&cfg);
    ASSERT(t != NULL, "trainer create");

    ce_status_t st = ce_trainer_step(t);
    ASSERT(st == CE_OK, "trainer step failed");

    ASSERT(g_forward_calls > 0, "probe forward never called");
    /* op_code occupies [2*max_bits, 2*max_bits+3) within the row */
    ASSERT(g_seen_width == 2 * max_bits + 3,
           "input row width must be 2*max_bits+3 (room for 3 op bits)");

    ce_trainer_destroy(t);
    ce_module_destroy(probe);
    PASS();
}

/* ---- Test: batch layout must follow the FINAL stage, not the current ----
 *
 * Tensors are allocated once for the final curriculum stage
 * (final=16 → input_dim = 2*16+3 = 35), but run_single_epoch used to pass
 * the CURRENT stage max (e.g. 8) as the layout anchor. With a narrower
 * current stage, b_bits/op_code land in the wrong columns and target/mask
 * rows are written with a too-small stride, overlapping each other.
 */
static void test_prepare_batch_final_anchor(void) {
    TEST("prepare: layout anchored to final stage");
    const int final_bits = 16;
    const int input_dim  = 2 * final_bits + 3;   /* 35 */

    float a0[] = {1,0,1,0};
    float b0[] = {0,1,0,1};
    float o0[] = {1,1,0};
    float r0[] = {1,0,1,1};
    float a1[] = {1,1,1,1,0,0,0,0};
    float b1[] = {1,0,1,0,1,0,1,0};
    float r1[] = {0,1,1,0,1,0,0,1};

    ce_alu_sample_t s0, s1;
    memset(&s0, 0, sizeof s0);
    memset(&s1, 0, sizeof s1);
    int64_t sh4[1] = {4}, sh8[1] = {8}, sh3[1] = {3};

    s0.a_bits      = ce_tensor_create_from_data(a0, sh4, 1);
    s0.b_bits      = ce_tensor_create_from_data(b0, sh4, 1);
    s0.op_code     = ce_tensor_create_from_data(o0, sh3, 1);
    s0.result_bits = ce_tensor_create_from_data(r0, sh4, 1);
    s0.n_bits      = 4;

    s1.a_bits      = ce_tensor_create_from_data(a1, sh8, 1);
    s1.b_bits      = ce_tensor_create_from_data(b1, sh8, 1);
    s1.op_code     = ce_tensor_create_from_data(o0, sh3, 1);
    s1.result_bits = ce_tensor_create_from_data(r1, sh8, 1);
    s1.n_bits      = 8;

    ce_alu_sample_t* samples[2] = {&s0, &s1};
    ce_batch_t batch;
    memset(&batch, 0, sizeof batch);
    batch.samples  = samples;
    batch.count    = 2;
    batch.capacity = 2;

    int64_t in_sh[2]  = {2, input_dim};
    int64_t tgt_sh[2] = {2, final_bits};
    ce_tensor_t* input  = ce_tensor_zeros(in_sh, 2);
    ce_tensor_t* target = ce_tensor_zeros(tgt_sh, 2);
    ce_tensor_t* mask   = ce_tensor_zeros(tgt_sh, 2);
    ASSERT(input && target && mask, "tensor alloc failed");

    ce_trainer_prepare_batch(&batch, input, target, mask, input_dim);

    /* Row 0 (nb=4): a at [0,4), zero pad to final, b at [16,20), op at [32,35) */
    for (int j = 0; j < 4; j++) {
        ASSERT(input->data[j] == a0[j], "row0 a_bits misplaced");
        ASSERT(input->data[final_bits + j] == b0[j],
               "row0 b_bits not at final-stage offset");
    }
    for (int j = 4; j < final_bits; j++)
        ASSERT(input->data[j] == 0.0f,
               "row0 a-padding not zeroed (b_bits leaked into a-block)");
    for (int j = 0; j < 3; j++)
        ASSERT(input->data[2 * final_bits + j] == o0[j],
               "row0 op_code not at 2*final offset");

    /* Row 1 (nb=8): same layout shifted by input_dim */
    const int base1 = input_dim;
    for (int j = 0; j < 8; j++) {
        ASSERT(input->data[base1 + j] == a1[j], "row1 a_bits misplaced");
        ASSERT(input->data[base1 + final_bits + j] == b1[j],
               "row1 b_bits not at final-stage offset");
    }
    for (int j = 8; j < final_bits; j++)
        ASSERT(input->data[base1 + j] == 0.0f,
               "row1 a-padding not zeroed (b_bits leaked into a-block)");
    for (int j = 0; j < 3; j++)
        ASSERT(input->data[base1 + 2 * final_bits + j] == o0[j],
               "row1 op_code not at 2*final offset");

    /* Target rows stride = final: row1 starts at final_bits, not nb/current */
    for (int j = 0; j < 4; j++)
        ASSERT(target->data[j] == r0[j], "target row0 result misplaced");
    for (int j = 4; j < final_bits; j++)
        ASSERT(target->data[j] == 0.0f, "target row0 padding not zeroed");
    for (int j = 0; j < 8; j++)
        ASSERT(target->data[final_bits + j] == r1[j],
               "target row1 not at final-stage stride");

    /* Mask rows follow the same stride */
    for (int j = 0; j < 4; j++)
        ASSERT(mask->data[j] == 1.0f, "mask row0 valid bits");
    for (int j = 4; j < final_bits; j++)
        ASSERT(mask->data[j] == 0.0f, "mask row0 padding");
    for (int j = 0; j < 8; j++)
        ASSERT(mask->data[final_bits + j] == 1.0f,
               "mask row1 not at final-stage stride");

    /* Sample tensors live for the duration of the test (process exits after). */
    PASS();
}

/* ---- Test: an epoch with zero processed batches must not fabricate loss ----
 *
 * run_single_epoch() used to compute avg_loss = 0.0f when actual_b == 0
 * (stop requested before the first batch, or samples_per_epoch <
 * batch_size leaving zero batches per epoch) and feed it everywhere:
 *   - ce_curriculum_update(0.0): 0.0 < loss_threshold counted as a
 *     "good" epoch, and with patience it advanced the stage with no
 *     training at all;
 *   - t->current_loss: clobbered the last real loss with a fake 0.0;
 *   - on_epoch_end / checkpoint: reported a perfect loss.
 * An epoch that processed no batches carries no signal — all three must
 * be skipped.
 */
static int g_cb_calls = 0;
static void count_epoch_cb(int epoch, float loss, void* user_data) {
    (void)epoch; (void)loss; (void)user_data;
    g_cb_calls++;
}

static void test_empty_epoch_no_fake_loss_or_advance(void) {
    TEST("empty epoch: no fake loss 0.0, no curriculum advance");

    ce_module_t* probe = make_probe_module();
    ASSERT(probe != NULL, "probe module create");

    ce_trainer_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.model               = probe;
    cfg.loss_type           = CE_LOSS_BCE;
    cfg.optimizer_type      = CE_OPT_ADAM;
    cfg.learning_rate       = 1e-3f;
    cfg.total_epochs        = 5;
    cfg.batch_size          = 4;
    cfg.samples_per_epoch   = 8;      /* → 2 batches per epoch */
    cfg.seed                = 42;
    cfg.max_bits            = 8;
    cfg.loss_threshold      = 0.5f;   /* probe loss ≈ ln2 ≈ 0.693 → "bad" */
    cfg.patience            = 1;      /* one fake 0.0 would advance */
    cfg.max_epochs_per_stage = 100;
    cfg.on_epoch_end        = count_epoch_cb;
    cfg.user_data           = NULL;

    g_cb_calls = 0;

    ce_trainer_t* t = ce_trainer_create(&cfg);
    ASSERT(t != NULL, "trainer create");

    /* Epoch 1: real batches → real loss, curriculum sees it. */
    ce_status_t st = ce_trainer_step(t);
    ASSERT(st == CE_OK, "first step failed");
    float real_loss = ce_trainer_current_loss(t);
    ASSERT(real_loss > 0.5f, "first epoch should report a real (non-zero) loss");
    ASSERT(ce_trainer_curriculum_stage(t) == 0,
           "real loss above threshold must not advance the stage");
    ASSERT(g_cb_calls == 1, "callback should fire for the real epoch");

    /* Epoch 2: stop before any batch → zero processed batches. */
    st = ce_trainer_stop(t);
    ASSERT(st == CE_OK, "stop failed");
    st = ce_trainer_step(t);
    ASSERT(st == CE_OK, "second step failed");

    ASSERT(ce_trainer_curriculum_stage(t) == 0,
           "empty epoch must not advance the curriculum stage");
    ASSERT(ce_trainer_current_loss(t) == real_loss,
           "empty epoch must not clobber the last real loss with 0.0");
    ASSERT(g_cb_calls == 1,
           "callback must not fire for an epoch that processed no batches");

    ce_trainer_destroy(t);
    ce_module_destroy(probe);
    PASS();
}

/* ---- Test: config with zero batches per epoch must be rejected ----
 *
 * samples_per_epoch < batch_size makes batches_per_epoch = 0, so every
 * epoch processes nothing — the trainer can only spin through
 * total_epochs fabricating losses (the root cause guarded above).  Such
 * a config is rejected at creation instead.
 */
static void test_create_rejects_sub_batch_epoch(void) {
    TEST("create rejects samples_per_epoch < batch_size");

    ce_module_t* probe = make_probe_module();
    ASSERT(probe != NULL, "probe module create");

    ce_trainer_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.model              = probe;
    cfg.loss_type          = CE_LOSS_BCE;
    cfg.optimizer_type     = CE_OPT_ADAM;
    cfg.learning_rate      = 1e-3f;
    cfg.total_epochs       = 1;
    cfg.batch_size         = 8;
    cfg.samples_per_epoch  = 4;   /* 4 < 8 → zero batches per epoch */
    cfg.seed               = 42;
    cfg.max_bits           = 8;
    cfg.loss_threshold     = 0.5f;
    cfg.patience           = 1;
    cfg.max_epochs_per_stage = 100;

    ce_trainer_t* t = ce_trainer_create(&cfg);
    ASSERT(t == NULL,
           "samples_per_epoch < batch_size yields zero batches per epoch "
           "and must be rejected");

    if (t) ce_trainer_destroy(t);
    ce_module_destroy(probe);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Trainer Tests ===\n\n");
    ce_init();

    test_trainer_op_code_layout();
    test_prepare_batch_final_anchor();
    test_empty_epoch_no_fake_loss_or_advance();
    test_create_rejects_sub_batch_epoch();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
