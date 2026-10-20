#ifndef CENGINE_TELEMETRY_H
#define CENGINE_TELEMETRY_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "cengine.h"
#include "memory.h"
#include "threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Metric Types
 * ======================================================================== */

typedef enum {
    /* Loss components */
    CE_METRIC_LOSS_TOTAL = 0,
    CE_METRIC_LOSS_RESULT,
    CE_METRIC_LOSS_CARRY,
    CE_METRIC_LOSS_BALANCE,
    CE_METRIC_LOSS_Z,

    /* Training hyperparams */
    CE_METRIC_LR,
    CE_METRIC_TEMPERATURE,

    /* Memory */
    CE_METRIC_MEM_CURRENT,
    CE_METRIC_MEM_PEAK,
    CE_METRIC_MEM_GROWTH_RATE,

    /* Gradients */
    CE_METRIC_GRAD_NORM,

    /* Expert / MoE */
    CE_METRIC_EXPERT_USAGE,     /* f32_arr[4] */

    /* Throughput */
    CE_METRIC_THROUGHPUT,

    /* Timing (ms) */
    CE_METRIC_EPOCH_TIME_MS,
    CE_METRIC_FORWARD_TIME_MS,
    CE_METRIC_BACKWARD_TIME_MS,
    CE_METRIC_OPTIMIZER_TIME_MS,
    CE_METRIC_DATA_GEN_TIME_MS,

    /* System */
    CE_METRIC_CPU_UTIL,

    /* Hardware perf counters */
    CE_METRIC_HW_IPC,
    CE_METRIC_HW_L2_MISS_RATE,
    CE_METRIC_HW_LLC_MISS_RATE,
    CE_METRIC_HW_BRANCH_MISS_RATE,

    /* Thread pool */
    CE_METRIC_ACTIVE_WORKERS,
    CE_METRIC_THREADPOOL_QUEUE_DEPTH,

    /* Anomaly detection */
    CE_METRIC_NAN_COUNT,
    CE_METRIC_INF_COUNT,

    /* Curriculum */
    CE_METRIC_CURRICULUM_STAGE,

    CE_METRIC_COUNT  /* sentinel */
} ce_metric_type_t;

/* Metric flags */
#define CE_METRIC_FLAG_VALID   0x01
#define CE_METRIC_FLAG_ALERT   0x02

/* ========================================================================
 * Telemetry Collector
 * ======================================================================== */

typedef struct {
    ce_ring_buffer_t* ring;

    /* Timing */
    struct timespec epoch_start;
    struct timespec forward_start;
    struct timespec backward_start;
    struct timespec optimizer_start;
    struct timespec data_gen_start;

    /* Counters */
    int64_t nan_count;
    int64_t inf_count;
    int     current_epoch;
    int64_t samples_processed;
} ce_collector_t;

ce_collector_t* ce_collector_create(ce_ring_buffer_t* ring);
void            ce_collector_destroy(ce_collector_t* c);

void ce_collector_start_epoch(ce_collector_t* c, int epoch);
void ce_collector_end_epoch(ce_collector_t* c);

void ce_collector_start_forward(ce_collector_t* c);
void ce_collector_end_forward(ce_collector_t* c);

void ce_collector_start_backward(ce_collector_t* c);
void ce_collector_end_backward(ce_collector_t* c);

void ce_collector_start_optimizer(ce_collector_t* c);
void ce_collector_end_optimizer(ce_collector_t* c);

void ce_collector_start_data_gen(ce_collector_t* c);
void ce_collector_end_data_gen(ce_collector_t* c);

void ce_collector_record_loss(ce_collector_t* c,
                              float total, float result,
                              float carry, float balance, float z);
void ce_collector_record_lr(ce_collector_t* c, float lr, float temp);
void ce_collector_record_memory(ce_collector_t* c,
                                const ce_memory_stats_t* stats);
void ce_collector_record_grad_norm(ce_collector_t* c, float norm);
void ce_collector_record_expert_usage(ce_collector_t* c, float usage[4]);
void ce_collector_record_throughput(ce_collector_t* c, float samples_per_sec);
void ce_collector_record_anomalies(ce_collector_t* c, int64_t nan_count,
                                   int64_t inf_count);
void ce_collector_record_curriculum(ce_collector_t* c, int stage);

/* ========================================================================
 * Telemetry Reporter
 * ======================================================================== */

/* Snapshot of all metrics for Python to poll */
typedef struct {
    float    loss_total;
    float    loss_result;
    float    loss_carry;
    float    loss_balance;
    float    loss_z;
    float    lr;
    float    temperature;
    float    mem_current_mb;
    float    mem_peak_mb;
    float    mem_growth_rate;
    float    grad_norm;
    float    expert_usage[4];
    float    throughput;
    float    epoch_time_ms;
    float    forward_time_ms;
    float    backward_time_ms;
    float    optimizer_time_ms;
    float    data_gen_time_ms;
    float    cpu_util;
    float    hw_ipc;
    float    hw_l2_miss_rate;
    float    hw_llc_miss_rate;
    float    hw_branch_miss_rate;
    int      active_workers;
    int      threadpool_queue_depth;
    int64_t  nan_count;
    int64_t  inf_count;
    int      curriculum_stage;
    int      current_epoch;
    uint64_t last_update_ns;
} ce_telemetry_snapshot_t;

typedef struct {
    ce_ring_buffer_t* ring;
    ce_telemetry_snapshot_t snapshot;
} ce_reporter_t;

ce_reporter_t* ce_reporter_create(ce_ring_buffer_t* ring);
void           ce_reporter_destroy(ce_reporter_t* r);

/* Drain ring buffer and update snapshot. Returns number of metrics consumed. */
int  ce_reporter_poll(ce_reporter_t* r);

/* Get a copy of the latest snapshot (thread-safe for readers) */
ce_telemetry_snapshot_t ce_reporter_get_snapshot(const ce_reporter_t* r);

/* ========================================================================
 * Hardware Performance Counters (perf_event_open)
 * ======================================================================== */

typedef struct {
    int fd_instructions;
    int fd_cycles;
    int fd_cache_references;
    int fd_cache_misses;
    int fd_branch_instructions;
    int fd_branch_misses;
    bool available;
} ce_perf_t;

ce_perf_t* ce_perf_create(void);
void       ce_perf_destroy(ce_perf_t* p);

void ce_perf_read(ce_perf_t* p, float* ipc,
                  float* l2_miss_rate, float* llc_miss_rate,
                  float* branch_miss_rate);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_TELEMETRY_H */
