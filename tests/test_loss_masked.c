/*
 * test_loss_masked.c — masked MSE loss/backward tests.
 *
 * Regression for a major bug (A3): ce_mse_loss / ce_mse_backward had no
 * mask parameter, so the MSE training path (trainer.c) computed loss and
 * gradients over padded positions too — unlike the BCE path, which honours
 * the batch mask.  The signatures now mirror BCE: an optional mask whose
 * zero entries are excluded from the mean and get zero gradient.
 *
 * The last test wires the whole path: it runs a trainer epoch with
 * CE_LOSS_MSE (probe model outputs zeros) and checks the reported loss
 * against masked MSE computed on the exact batch stream the trainer
 * consumes — the unmasked value differs whenever padding exists.
 */
#include "cengine.h"
#include "data.h"
#include "loss.h"
#include "tensor.h"
#include "trainer.h"
#include "../src/training/trainer_internal.h"
#include <math.h>
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

static ce_tensor_t* make10(const float* data, int64_t n) {
    int64_t shape[1] = { n };
    return ce_tensor_create_from_data((float*)data, shape, 1);
}

/* ---- Masked loss excludes padded elements from the mean ---- */
static void test_mse_loss_masked(void) {
    TEST("mse loss: mask excludes padded, mean over valid");
    float p[] = { 1, 3, 1, 3 };       /* pred */
    float t[] = { 1, 0, 1, 0 };       /* target: 1s only at valid pos */
    float m[] = { 1, 0, 1, 0 };       /* mask */

    ce_tensor_t* pred  = make10(p, 4);
    ce_tensor_t* tgt   = make10(t, 4);
    ce_tensor_t* mask  = make10(m, 4);
    ASSERT(pred && tgt && mask, "tensor alloc");

    /* valid: (1-1)^2 + (1-1)^2 = 0 → mean 0;
     * unmasked: (0 + 9 + 0 + 9) / 4 = 4.5 */
    float masked = ce_mse_loss(pred, tgt, mask);
    ASSERT(masked == 0.0f,
           "padded positions leaked into masked MSE (expected 0)");

    /* unmasked (NULL) keeps legacy semantics: mean over all n */
    float plain = ce_mse_loss(pred, tgt, NULL);
    ASSERT(fabsf(plain - 4.5f) < 1e-6f,
           "NULL-mask MSE must keep dividing by full numel");

    ce_tensor_release(pred);
    ce_tensor_release(tgt);
    ce_tensor_release(mask);
    PASS();
}

/* ---- Masked backward: zero grad on padding, scaled by n_valid ---- */
static void test_mse_backward_masked(void) {
    TEST("mse backward: zero grad on pad, scale by n_valid");
    float p[] = { 2, 2, 2, 2, 2, 2, 2, 2, 9, 9 };  /* 8 AVX + 2 tail */
    float t[] = { 1, 0, 0, 0, 0, 0, 0, 0, 1, 1 };
    float m[] = { 1, 0, 0, 0, 0, 0, 0, 0, 1, 1 };  /* n_valid = 3 */

    ce_tensor_t* pred = make10(p, 10);
    ce_tensor_t* tgt  = make10(t, 10);
    ce_tensor_t* mask = make10(m, 10);
    float g_init[] = { -7, -7, -7, -7, -7, -7, -7, -7, -7, -7 };
    ce_tensor_t* grad = make10(g_init, 10);
    ASSERT(pred && tgt && mask && grad, "tensor alloc");

    ce_status_t st = ce_mse_backward(pred, tgt, mask, grad);
    ASSERT(st == CE_OK, "masked backward status");

    /* grad = 2*(pred-target)/n_valid at valid pos, 0 at padded pos */
    float expect0 = 2.0f * (2.0f - 1.0f) / 3.0f;   /* index 0: p=2, t=1 */
    float expect8 = 2.0f * (9.0f - 1.0f) / 3.0f;   /* index 8: p=9, t=1 */
    ASSERT(fabsf(grad->data[0] - expect0) < 1e-6f,
           "valid-position gradient wrong (n_valid scaling)");
    for (int j = 1; j < 8; j++) {
        ASSERT(grad->data[j] == 0.0f,
               "padded position got a non-zero gradient");
    }
    ASSERT(fabsf(grad->data[8] - expect8) < 1e-6f,
           "AVX tail valid gradient wrong");

    ce_tensor_release(pred);
    ce_tensor_release(tgt);
    ce_tensor_release(mask);
    ce_tensor_release(grad);
    PASS();
}

/* ---- Guards mirror BCE ---- */
static void test_mse_masked_guards(void) {
    TEST("mse masked: shape guards mirror BCE");
    float a[] = { 1, 2 };
    float b[] = { 1, 2, 3 };
    float m[] = { 1, 1, 1 };

    ce_tensor_t* x2 = make10(a, 2);
    ce_tensor_t* x3 = make10(b, 3);
    ce_tensor_t* m3 = make10(m, 3);
    ASSERT(x2 && x3 && m3, "tensor alloc");

    ASSERT(ce_mse_loss(NULL, x2, NULL) == 0.0f, "NULL pred loss guard");
    ASSERT(ce_mse_loss(x2, x3, NULL) == 0.0f, "numel mismatch loss guard");
    ASSERT(ce_mse_loss(x2, x2, m3) == 0.0f, "mask numel mismatch loss guard");

    ce_tensor_t* grad = make10(a, 2);
    ASSERT(grad, "grad alloc");
    ASSERT(ce_mse_backward(NULL, x2, NULL, grad) == CE_ERR_INVALID_ARG,
           "NULL pred backward guard");
    ASSERT(ce_mse_backward(x2, x3, NULL, grad) == CE_ERR_INVALID_SHAPE,
           "numel mismatch backward guard");
    ASSERT(ce_mse_backward(x2, x2, m3, grad) == CE_ERR_INVALID_SHAPE,
           "mask mismatch backward guard");

    ce_tensor_release(x2);
    ce_tensor_release(x3);
    ce_tensor_release(m3);
    ce_tensor_release(grad);
    PASS();
}

/* =====================================================================
 * Trainer integration: MSE path must apply the batch mask.
 * Probe model outputs zeros → per-element loss = target^2, so
 * masked vs unmasked differ exactly when the batch has padding.
 * ===================================================================== */

static float g_cb_loss = -1.0f;
static int   g_cb_calls = 0;
static void record_loss(int epoch, float loss, void* user_data) {
    (void)epoch; (void)user_data;
    g_cb_loss = loss;
    g_cb_calls++;
}

static ce_status_t zero_forward(struct ce_module* self,
                                ce_tensor_t* input, ce_tensor_t* output) {
    (void)self; (void)input;
    ce_tensor_zero(output);
    return CE_OK;
}
static ce_status_t zero_backward(struct ce_module* self,
                                 ce_tensor_t* g_out, ce_tensor_t* g_in) {
    (void)self; (void)g_out; (void)g_in;
    return CE_OK;
}

static void test_trainer_mse_uses_mask(void) {
    TEST("trainer MSE path applies the batch mask");

    const int max_bits = 8;
    const int input_dim = 2 * max_bits + 3;
    const int batch_size = 4;
    const int samples_per_epoch = 8;   /* → 2 batches per epoch */
    const int64_t seed = 42;

    ce_module_t* probe = ce_module_create("zero-probe");
    ASSERT(probe != NULL, "probe create");
    int64_t pshape[1] = { 1 };
    ce_tensor_t* p = ce_tensor_zeros(pshape, 1);
    ce_tensor_t* g = ce_tensor_zeros(pshape, 1);
    ASSERT(p && g && ce_module_add_parameter(probe, p, g) == CE_OK,
           "probe parameter");
    probe->forward  = zero_forward;
    probe->backward  = zero_backward;
    probe->zero_grad = ce_module_default_zero_grad;

    ce_trainer_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.model               = probe;
    cfg.loss_type           = CE_LOSS_MSE;
    cfg.optimizer_type      = CE_OPT_ADAM;
    cfg.learning_rate       = 1e-3f;
    cfg.total_epochs        = 1;
    cfg.batch_size          = batch_size;
    cfg.samples_per_epoch   = samples_per_epoch;
    cfg.seed                = seed;
    cfg.max_bits            = max_bits;
    cfg.loss_threshold      = 0.0f;   /* never advance */
    cfg.patience            = 1;
    cfg.max_epochs_per_stage = 100;
    cfg.on_epoch_end        = record_loss;
    cfg.user_data           = NULL;

    g_cb_loss = -1.0f;
    g_cb_calls = 0;

    ce_trainer_t* t = ce_trainer_create(&cfg);
    ASSERT(t != NULL, "trainer create");
    ce_status_t st = ce_trainer_step(t);
    ASSERT(st == CE_OK, "trainer step");
    ASSERT(g_cb_calls == 1, "callback should fire for a real epoch");

    /* Replay the trainer's exact batch stream: same curriculum, same
     * seed → identical batches.  Probe outputs zeros, so masked MSE on
     * (zero, target, mask) is the expected reported loss. */
    ce_curriculum_t* cur = ce_curriculum_create(0.0f, 1, 100, max_bits);
    ASSERT(cur != NULL, "replica curriculum");
    ce_alu_gen_t* gen = ce_curriculum_make_generator(cur, seed);
    ASSERT(gen != NULL, "replica generator");

    int64_t in_shape[2]  = { batch_size, input_dim };
    int64_t out_shape[2] = { batch_size, max_bits };
    ce_tensor_t* input  = ce_tensor_zeros(in_shape, 2);
    ce_tensor_t* target = ce_tensor_zeros(out_shape, 2);
    ce_tensor_t* mask   = ce_tensor_zeros(out_shape, 2);
    ce_tensor_t* zeros  = ce_tensor_zeros(out_shape, 2);
    ASSERT(input && target && mask && zeros, "replica tensors");

    int n_batches = samples_per_epoch / batch_size;
    double masked_sum = 0.0, unmasked_sum = 0.0;
    int padded = 0;
    for (int b = 0; b < n_batches; b++) {
        ce_batch_t* batch = ce_alu_gen_batch(gen, batch_size);
        ASSERT(batch != NULL, "replica batch");
        ce_trainer_prepare_batch(batch, input, target, mask, input_dim);
        masked_sum   += ce_mse_loss(zeros, target, mask);
        unmasked_sum += ce_mse_loss(zeros, target, NULL);
        for (int i = 0; i < batch_size; i++) {
            if (batch->samples[i]->n_bits < max_bits) padded = 1;
        }
        ce_batch_destroy(batch);
    }
    double expected_masked   = masked_sum / n_batches;
    double expected_unmasked = unmasked_sum / n_batches;

    ASSERT(padded, "seed must yield padding (else the test proves nothing)");
    ASSERT(fabs(expected_unmasked - expected_masked) > 1e-9,
           "masked and unmasked expectations must differ when padded");
    ASSERT(fabs((double)g_cb_loss - expected_masked) < 1e-5,
           "trainer MSE loss does not match masked MSE on its own batch "
           "stream — trainer.c likely passes no mask");

    ce_tensor_release(input);
    ce_tensor_release(target);
    ce_tensor_release(mask);
    ce_tensor_release(zeros);
    ce_curriculum_destroy(cur);
    free(gen);
    ce_trainer_destroy(t);
    ce_module_destroy(probe);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Masked MSE Tests ===\n\n");
    ce_init();

    test_mse_loss_masked();
    test_mse_backward_masked();
    test_mse_masked_guards();
    test_trainer_mse_uses_mask();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
