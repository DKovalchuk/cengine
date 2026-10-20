#ifndef CENGINE_TRAINER_H
#define CENGINE_TRAINER_H

#include <stdbool.h>
#include <stdint.h>
#include "cengine.h"
#include "tensor.h"
#include "layers.h"
#include "autograd.h"
#include "optimizer.h"
#include "loss.h"
#include "data.h"
#include "threadpool.h"
#include "telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Loss type enumeration
 * ================================================================ */

typedef enum {
    CE_LOSS_BCE = 0,
    CE_LOSS_MSE,
} ce_loss_type_t;

/* ================================================================
 * Optimizer type enumeration
 * ================================================================ */

typedef enum {
    CE_OPT_ADAM = 0,
    CE_OPT_SGD,
} ce_optimizer_type_t;

/* ================================================================
 * Trainer configuration
 * ================================================================ */

typedef struct {
    /* Model */
    ce_module_t* model;

    /* Loss function type */
    ce_loss_type_t loss_type;

    /* Auxiliary loss weights (MoE ALU balance / z / carry) */
    float lambda_balance;
    float lambda_z;
    float lambda_carry;

    /* Optimizer */
    ce_optimizer_type_t optimizer_type;
    float learning_rate;
    float weight_decay;
    float beta1, beta2, eps;   /* Adam params */
    float momentum;            /* SGD param */

    /* Scheduler (OneCycle-style) */
    float lr_max, lr_min;
    float pct_start;
    int   total_epochs;

    /* Gradient clipping */
    float grad_clip_norm;

    /* Temperature (Gumbel-Softmax annealing) */
    float temp_start, temp_end;
    int   temp_anneal_epochs;

    /* Data */
    int     batch_size;
    int     samples_per_epoch;
    int64_t seed;

    /* Curriculum */
    float loss_threshold;
    int   patience;
    int   max_epochs_per_stage;
    int   max_bits;          /* cap curriculum stages to this max bit width */

    /* Threading */
    int num_workers;

    /* Telemetry */
    bool enable_telemetry;
    int  telemetry_interval;   /* emit metrics every N batches */

    /* Callback: called after each epoch with metrics */
    void (*on_epoch_end)(int epoch, float loss, void* user_data);
    void* user_data;

    /* Checkpoint */
    const char* checkpoint_dir;
    int         save_every_n_epochs;
} ce_trainer_config_t;

/* ================================================================
 * Trainer handle (opaque)
 * ================================================================ */

typedef struct ce_trainer ce_trainer_t;

/* Lifecycle */
ce_trainer_t*  ce_trainer_create(const ce_trainer_config_t* config);
ce_status_t    ce_trainer_destroy(ce_trainer_t* trainer);

/* Execution */
ce_status_t    ce_trainer_run(ce_trainer_t* trainer);   /* blocking: full training */
ce_status_t    ce_trainer_step(ce_trainer_t* trainer);  /* single epoch */
ce_status_t    ce_trainer_stop(ce_trainer_t* trainer);  /* graceful stop */

/* Accessors */
ce_ring_buffer_t* ce_trainer_get_telemetry(ce_trainer_t* trainer);

/* ================================================================
 * Checkpoint I/O
 * ================================================================ */

/* Save: binary file with header + per-parameter blobs */
ce_status_t  ce_checkpoint_save(const char* dir, int epoch, float loss,
                                ce_module_t* model, void* optimizer_state);

/* Load: restore weights into model (optimizer_state may be NULL) */
ce_status_t  ce_checkpoint_load(const char* dir, ce_module_t* model);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_TRAINER_H */
