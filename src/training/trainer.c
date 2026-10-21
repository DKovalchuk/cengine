/*
 * trainer.c — Main training loop with curriculum learning, telemetry,
 *             gradient clipping, and checkpoint support.
 *
 * Designed to be model-agnostic: calls model->forward/backward generically.
 * Data preparation is ALU-specific (concatenates a_bits, b_bits, op_code).
 */
#include "trainer.h"
#include "trainer_internal.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ====================================================================
 * Internal state
 * ==================================================================== */

struct ce_trainer {
    ce_trainer_config_t config;

    /* Flattened parameter/gradient arrays for optimizer */
    ce_tensor_t** all_params;
    ce_tensor_t** all_grads;
    int           n_all_params;

    /* Optimizer (exactly one is non-NULL) */
    ce_adam_t* adam;
    ce_sgd_t*  sgd;

    /* Scheduler */
    ce_scheduler_t* scheduler;

    /* Curriculum + data */
    ce_curriculum_t*  curriculum;
    ce_alu_gen_t*     generator;
    ce_dataloader_t*  dataloader;

    /* Telemetry (NULL when disabled) */
    ce_ring_buffer_t* ring;
    ce_collector_t*   collector;

    /* Pre-allocated working tensors (sized for the final curriculum stage) */
    ce_tensor_t* input_tensor;    /* [batch_size, input_dim] */
    ce_tensor_t* target_tensor;   /* [batch_size, max_bits]  */
    ce_tensor_t* output_tensor;   /* [batch_size, max_bits]  */
    ce_tensor_t* mask_tensor;     /* [batch_size, max_bits]  */
    ce_tensor_t* loss_grad_tensor;/* [batch_size, max_bits]  */
    int          max_bits;        /* final-stage max bits    */
    int          input_dim;       /* 2 * max_bits + 3        */

    /* Run state */
    int   current_epoch;
    float current_loss;
    float current_temperature;
    volatile bool stop_requested;
    bool  initialized;
};

/* ====================================================================
 * Helpers
 * ==================================================================== */

/* Recursively collect all parameter/gradient tensors from a module tree. */
static void collect_params_recursive(ce_module_t* mod,
                                     ce_tensor_t*** out_p,
                                     ce_tensor_t*** out_g,
                                     int* count, int* cap)
{
    if (!mod) return;

    for (int i = 0; i < mod->parameters->count; i++) {
        ce_tensor_t* p = mod->parameters->tensors[i];
        if (!p) continue;

        if (*count >= *cap) {
            *cap *= 2;
            *out_p = realloc(*out_p, (size_t)(*cap) * sizeof(ce_tensor_t*));
            *out_g = realloc(*out_g, (size_t)(*cap) * sizeof(ce_tensor_t*));
            if (!*out_p || !*out_g) return;
        }
        (*out_p)[*count] = p;
        (*out_g)[*count] = (i < mod->gradients->count) ? mod->gradients->tensors[i] : NULL;
        (*count)++;
    }

    /* Recurse into sequential children only (not linear/activation internal_state) */
    if (mod->forward == ce_sequential_forward && mod->internal_state) {
        ce_sequential_t* seq = (ce_sequential_t*)mod->internal_state;
        for (int i = 0; i < seq->count; i++) {
            collect_params_recursive(seq->children[i], out_p, out_g, count, cap);
        }
    }
}

/* Linear temperature annealing. */
static float compute_temperature(const ce_trainer_t* t)
{
    if (t->config.temp_anneal_epochs <= 0) return t->config.temp_start;
    if (t->current_epoch >= t->config.temp_anneal_epochs) return t->config.temp_end;
    float progress = (float)t->current_epoch / (float)t->config.temp_anneal_epochs;
    return t->config.temp_start
         + (t->config.temp_end - t->config.temp_start) * progress;
}

/* Destroy current dataloader + generator and rebuild for current stage. */
static ce_status_t rebuild_dataloader(ce_trainer_t* t)
{
    if (t->dataloader) { ce_dataloader_destroy(t->dataloader); t->dataloader = NULL; }
    if (t->generator)  { free(t->generator); t->generator = NULL; }

    t->generator = ce_curriculum_make_generator(t->curriculum,
                    t->config.seed + (int64_t)t->current_epoch);
    if (!t->generator) return CE_ERR_OUT_OF_MEMORY;

    t->dataloader = ce_dataloader_create(t->generator,
                    t->config.batch_size,
                    t->config.samples_per_epoch,
                    t->config.seed + (int64_t)t->current_epoch);
    if (!t->dataloader) return CE_ERR_OUT_OF_MEMORY;

    return CE_OK;
}

/*
 * Prepare input / target / mask tensors from an ALU batch.
 *
 * Input layout per row, anchored to the FINAL curriculum stage:
 *   [ a_bits(final) | b_bits(final) | op_code(3) ]
 * zero-padded up to input_dim, where final = (input_dim - 3) / 2.
 * target/mask rows use the same final stride.
 *
 * The anchor must be input_dim (the model's contract, allocated once for
 * the final stage) — not the current stage's max_bits: with a narrower
 * current stage, b_bits/op_code landed in the wrong columns and
 * target/mask rows overlapped, silently corrupting training data.
 *
 * The mask is 1.0 for bit positions [0, sample_n_bits) and 0.0 elsewhere,
 * so BCE loss ignores padded positions.
 */
void ce_trainer_prepare_batch(const ce_batch_t* batch,
                              ce_tensor_t* input, ce_tensor_t* target,
                              ce_tensor_t* mask, int input_dim)
{
    int bs = batch->count;
    const int final = (input_dim - 3) / 2;

    ce_tensor_zero(input);
    ce_tensor_zero(target);
    ce_tensor_zero(mask);

    for (int i = 0; i < bs; i++) {
        ce_alu_sample_t* s = batch->samples[i];
        int nb = s->n_bits;
        if (nb > final) nb = final;

        float* in_row  = &input->data[i * input_dim];
        float* tgt_row = &target->data[i * final];
        float* msk_row = &mask->data[i * final];

        /* a_bits */
        memcpy(in_row, s->a_bits->data, (size_t)nb * sizeof(float));
        /* b_bits */
        memcpy(in_row + final, s->b_bits->data, (size_t)nb * sizeof(float));
        /* op_code (always 3 floats) */
        memcpy(in_row + 2 * final, s->op_code->data, 3 * sizeof(float));

        /* target result_bits */
        memcpy(tgt_row, s->result_bits->data, (size_t)nb * sizeof(float));

        /* mask: 1 for valid positions */
        for (int j = 0; j < nb; j++) {
            msk_row[j] = 1.0f;
        }
    }
}

/* ====================================================================
 * One-time initialization (called lazily from run/step)
 * ==================================================================== */

static ce_status_t init_training(ce_trainer_t* t)
{
    if (t->initialized) return CE_OK;

    ce_status_t st;

    /* ---- Collect model parameters ---- */
    int param_cap = 16;
    t->all_params = malloc((size_t)param_cap * sizeof(ce_tensor_t*));
    t->all_grads  = malloc((size_t)param_cap * sizeof(ce_tensor_t*));
    if (!t->all_params || !t->all_grads) return CE_ERR_OUT_OF_MEMORY;

    t->n_all_params = 0;
    collect_params_recursive(t->config.model,
                             &t->all_params, &t->all_grads,
                             &t->n_all_params, &param_cap);

    if (t->n_all_params == 0) {
        fprintf(stderr, "ce_trainer: model has no parameters\n");
        return CE_ERR_NOT_INITIALIZED;
    }

    /* ---- Optimizer ---- */
    if (t->config.optimizer_type == CE_OPT_ADAM) {
        t->adam = ce_adam_create(
            t->config.learning_rate,
            t->config.beta1, t->config.beta2, t->config.eps,
            t->config.weight_decay,
            t->all_params, t->all_grads, t->n_all_params);
        if (!t->adam) return CE_ERR_OUT_OF_MEMORY;
    } else {
        t->sgd = ce_sgd_create(
            t->config.learning_rate,
            t->config.momentum, t->config.weight_decay,
            t->all_params, t->all_grads, t->n_all_params);
        if (!t->sgd) return CE_ERR_OUT_OF_MEMORY;
    }

    /* ---- Scheduler ---- */
    int batches_per_epoch = t->config.samples_per_epoch / t->config.batch_size;
    int total_steps = t->config.total_epochs * batches_per_epoch;

    if (t->config.lr_max > 0.0f && total_steps > 0) {
        float div_factor = (t->config.learning_rate > 0.0f)
            ? t->config.lr_max / t->config.learning_rate
            : 25.0f;
        float final_div_factor = (t->config.lr_min > 0.0f &&
                                  t->config.learning_rate > 0.0f)
            ? t->config.learning_rate / t->config.lr_min
            : 10000.0f;
        if (div_factor <= 0.0f) div_factor = 25.0f;
        if (final_div_factor <= 0.0f) final_div_factor = 10000.0f;

        t->scheduler = ce_scheduler_create(
            t->config.lr_max, t->config.pct_start,
            div_factor, final_div_factor, total_steps);
        if (!t->scheduler) return CE_ERR_OUT_OF_MEMORY;
    }

    /* ---- Curriculum ---- */
    t->curriculum = ce_curriculum_create(
        t->config.loss_threshold,
        t->config.patience,
        t->config.max_epochs_per_stage,
        t->config.max_bits);
    if (!t->curriculum) return CE_ERR_OUT_OF_MEMORY;

    /* ---- Generator + DataLoader ---- */
    st = rebuild_dataloader(t);
    if (st != CE_OK) return st;

    /* ---- Telemetry ---- */
    if (t->config.enable_telemetry) {
        t->ring = calloc(1, sizeof(ce_ring_buffer_t));
        if (!t->ring) return CE_ERR_OUT_OF_MEMORY;
        t->collector = ce_collector_create(t->ring);
        if (!t->collector) return CE_ERR_OUT_OF_MEMORY;
    }

    /* ---- Working tensors (sized for the final curriculum stage) ---- */
    int max_bits  = t->curriculum->stages[t->curriculum->n_stages - 1].max_bits;
    /* op_code is a 3-float binary encoding (data.h: [3]) — reserving only
     * 2 slots made prepare_batch write the third float past the row and,
     * for the last row of a batch, past the input buffer (heap OOB write). */
    int input_dim = 2 * max_bits + 3;
    int bs        = t->config.batch_size;

    int64_t in_shape[2]  = { bs, input_dim };
    int64_t out_shape[2] = { bs, max_bits };

    t->input_tensor     = ce_tensor_create(in_shape,  2, CE_FLOAT32);
    t->target_tensor    = ce_tensor_create(out_shape, 2, CE_FLOAT32);
    t->output_tensor    = ce_tensor_create(out_shape, 2, CE_FLOAT32);
    t->mask_tensor      = ce_tensor_create(out_shape, 2, CE_FLOAT32);
    t->loss_grad_tensor = ce_tensor_create(out_shape, 2, CE_FLOAT32);

    if (!t->input_tensor || !t->target_tensor || !t->output_tensor ||
        !t->mask_tensor  || !t->loss_grad_tensor)
        return CE_ERR_OUT_OF_MEMORY;

    t->max_bits  = max_bits;
    t->input_dim = input_dim;
    t->current_temperature = t->config.temp_start;

    t->initialized = true;
    return CE_OK;
}

/* ====================================================================
 * Single-epoch body (shared by run and step)
 * ==================================================================== */

static ce_status_t run_single_epoch(ce_trainer_t* t)
{
    int epoch = t->current_epoch;
    int batches_per_epoch = t->config.samples_per_epoch / t->config.batch_size;

    if (t->collector) ce_collector_start_epoch(t->collector, epoch);

    double total_loss  = 0.0;
    int    actual_b    = 0;

    for (int b = 0; b < batches_per_epoch; b++) {
        if (t->stop_requested) break;

        /* ---- Data generation ---- */
        if (t->collector) ce_collector_start_data_gen(t->collector);
        ce_batch_t* batch = ce_dataloader_next(t->dataloader);
        if (t->collector) ce_collector_end_data_gen(t->collector);
        if (!batch || batch->count == 0) {
            if (batch) ce_batch_destroy(batch);
            continue;
        }

        /* ---- Prepare tensors ---- */
        ce_trainer_prepare_batch(batch,
                      t->input_tensor, t->target_tensor, t->mask_tensor,
                      t->input_dim);

        /* ---- Zero gradients ---- */
        t->config.model->zero_grad(t->config.model);

        /* ---- Forward pass ---- */
        if (t->collector) ce_collector_start_forward(t->collector);
        ce_status_t st = t->config.model->forward(
            t->config.model, t->input_tensor, t->output_tensor);
        if (t->collector) ce_collector_end_forward(t->collector);
        if (st != CE_OK) { ce_batch_destroy(batch); return st; }

        /* ---- Compute loss + gradient ---- */
        float loss;
        if (t->config.loss_type == CE_LOSS_BCE) {
            loss = ce_bce_with_logits_loss(
                t->output_tensor, t->target_tensor, t->mask_tensor);
            ce_bce_with_logits_backward(
                t->output_tensor, t->target_tensor,
                t->mask_tensor, t->loss_grad_tensor);
        } else {
            loss = ce_mse_loss(t->output_tensor, t->target_tensor);
            ce_mse_backward(
                t->output_tensor, t->target_tensor, t->loss_grad_tensor);
        }

        total_loss += (double)loss;
        actual_b++;

        /* ---- Backward pass ---- */
        if (t->collector) ce_collector_start_backward(t->collector);
        st = t->config.model->backward(
            t->config.model, t->loss_grad_tensor, NULL);
        if (t->collector) ce_collector_end_backward(t->collector);
        if (st != CE_OK) { ce_batch_destroy(batch); return st; }

        /* ---- Gradient clipping ---- */
        float grad_norm = ce_compute_grad_norm(t->all_grads, t->n_all_params);
        if (t->config.grad_clip_norm > 0.0f) {
            ce_clip_grad_norm(t->all_grads, t->n_all_params,
                              t->config.grad_clip_norm);
        }

        /* ---- Optimizer step ---- */
        if (t->collector) ce_collector_start_optimizer(t->collector);

        float lr;
        if (t->scheduler) {
            lr = ce_scheduler_step(t->scheduler);
        } else {
            lr = t->config.learning_rate;
        }

        if (t->adam) { t->adam->lr = lr; ce_adam_step(t->adam); }
        if (t->sgd)  { t->sgd->lr  = lr; ce_sgd_step(t->sgd);  }

        if (t->collector) ce_collector_end_optimizer(t->collector);

        /* ---- Telemetry ---- */
        if (t->collector &&
            t->config.telemetry_interval > 0 &&
            b % t->config.telemetry_interval == 0)
        {
            ce_collector_record_loss(t->collector,
                                     loss, loss, 0.0f, 0.0f, 0.0f);
            ce_collector_record_lr(t->collector, lr,
                                   t->current_temperature);
            ce_collector_record_grad_norm(t->collector, grad_norm);

            ce_memory_stats_t mem = ce_tracker_get_stats();
            ce_collector_record_memory(t->collector, &mem);
        }

        ce_batch_destroy(batch);
    }

    /* ---- End-of-epoch bookkeeping ---- */
    float avg_loss = (actual_b > 0)
        ? (float)(total_loss / (double)actual_b) : 0.0f;
    t->current_loss = avg_loss;

    if (t->collector) ce_collector_end_epoch(t->collector);

    /* Temperature annealing */
    t->current_temperature = compute_temperature(t);

    /* Curriculum update — may advance to next stage */
    int advanced = ce_curriculum_update(t->curriculum, avg_loss);
    if (advanced) {
        ce_status_t st = rebuild_dataloader(t);
        if (st != CE_OK) return st;
        if (t->collector) {
            ce_collector_record_curriculum(
                t->collector, t->curriculum->current_stage);
        }
    }

    /* Periodic checkpoint */
    if (t->config.checkpoint_dir && t->config.checkpoint_dir[0] &&
        t->config.save_every_n_epochs > 0 &&
        (epoch + 1) % t->config.save_every_n_epochs == 0)
    {
        ce_checkpoint_save(t->config.checkpoint_dir, epoch, avg_loss,
                           t->config.model, NULL);
    }

    /* User callback */
    if (t->config.on_epoch_end) {
        t->config.on_epoch_end(epoch, avg_loss, t->config.user_data);
    }

    t->current_epoch = epoch + 1;
    return CE_OK;
}

/* ====================================================================
 * Public API
 * ==================================================================== */

ce_trainer_t* ce_trainer_create(const ce_trainer_config_t* config)
{
    if (!config || !config->model) return NULL;
    if (config->batch_size <= 0 || config->samples_per_epoch <= 0) return NULL;
    if (config->total_epochs <= 0) return NULL;

    ce_trainer_t* t = calloc(1, sizeof(*t));
    if (!t) return NULL;

    t->config = *config;
    return t;
}

ce_status_t ce_trainer_run(ce_trainer_t* t)
{
    if (!t) return CE_ERR_INVALID_ARG;

    ce_status_t st = init_training(t);
    if (st != CE_OK) return st;

    for (int e = t->current_epoch; e < t->config.total_epochs; e++) {
        if (t->stop_requested) break;

        t->current_epoch = e;
        st = run_single_epoch(t);
        if (st != CE_OK) return st;
    }

    /* Final checkpoint */
    if (t->config.checkpoint_dir && t->config.checkpoint_dir[0]) {
        ce_checkpoint_save(t->config.checkpoint_dir,
                           t->current_epoch, t->current_loss,
                           t->config.model, NULL);
    }

    return CE_OK;
}

ce_status_t ce_trainer_step(ce_trainer_t* t)
{
    if (!t) return CE_ERR_INVALID_ARG;

    ce_status_t st = init_training(t);
    if (st != CE_OK) return st;

    if (t->current_epoch >= t->config.total_epochs) return CE_OK;

    return run_single_epoch(t);
}

ce_status_t ce_trainer_stop(ce_trainer_t* t)
{
    if (!t) return CE_ERR_INVALID_ARG;
    t->stop_requested = true;
    return CE_OK;
}

ce_ring_buffer_t* ce_trainer_get_telemetry(ce_trainer_t* t)
{
    return t ? t->ring : NULL;
}

ce_status_t ce_trainer_destroy(ce_trainer_t* t)
{
    if (!t) return CE_OK;

    /* Working tensors */
    if (t->input_tensor)     ce_tensor_release(t->input_tensor);
    if (t->target_tensor)    ce_tensor_release(t->target_tensor);
    if (t->output_tensor)    ce_tensor_release(t->output_tensor);
    if (t->mask_tensor)      ce_tensor_release(t->mask_tensor);
    if (t->loss_grad_tensor) ce_tensor_release(t->loss_grad_tensor);

    /* Telemetry */
    if (t->collector) ce_collector_destroy(t->collector);
    free(t->ring);

    /* Data */
    if (t->dataloader) ce_dataloader_destroy(t->dataloader);
    free(t->generator);

    /* Curriculum */
    if (t->curriculum) ce_curriculum_destroy(t->curriculum);

    /* Scheduler */
    if (t->scheduler) ce_scheduler_destroy(t->scheduler);

    /* Optimizer */
    if (t->adam) ce_adam_destroy(t->adam);
    if (t->sgd)  ce_sgd_destroy(t->sgd);

    /* Parameter arrays */
    free(t->all_params);
    free(t->all_grads);

    free(t);
    return CE_OK;
}
