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

int main(void) {
    printf("\n=== CEngine Trainer Tests ===\n\n");
    ce_init();

    test_trainer_op_code_layout();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
