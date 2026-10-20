#include "cengine.h"
#include "memory.h"
#include "telemetry.h"

#include <stdlib.h>
#include <string.h>

/* ---- Helpers ---- */

static uint64_t timespec_to_ns(const struct timespec* ts) {
    return (uint64_t)ts->tv_sec * 1000000000ULL + (uint64_t)ts->tv_nsec;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return timespec_to_ns(&ts);
}

static float elapsed_ms(const struct timespec* start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    float sec = (float)(now.tv_sec - start->tv_sec);
    float nsec = (float)(now.tv_nsec - start->tv_nsec);
    return sec * 1000.0f + nsec / 1e6f;
}

static void emit_f32(ce_ring_buffer_t* ring, uint8_t type,
                     uint32_t epoch, float value) {
    ce_metric_t m;
    memset(&m, 0, sizeof(m));
    m.metric_type = type;
    m.flags = CE_METRIC_FLAG_VALID;
    m.epoch = epoch;
    m.timestamp_ns = now_ns();
    m.value.f32 = value;
    ce_ring_push(ring, &m);
}

static void emit_i64(ce_ring_buffer_t* ring, uint8_t type,
                     uint32_t epoch, int64_t value) {
    ce_metric_t m;
    memset(&m, 0, sizeof(m));
    m.metric_type = type;
    m.flags = CE_METRIC_FLAG_VALID;
    m.epoch = epoch;
    m.timestamp_ns = now_ns();
    m.value.i64 = value;
    ce_ring_push(ring, &m);
}

static void emit_f32_arr(ce_ring_buffer_t* ring, uint8_t type,
                         uint32_t epoch, const float* arr, int count) {
    ce_metric_t m;
    memset(&m, 0, sizeof(m));
    m.metric_type = type;
    m.flags = CE_METRIC_FLAG_VALID;
    m.epoch = epoch;
    m.timestamp_ns = now_ns();
    int n = (count > 8) ? 8 : count;
    memcpy(m.value.f32_arr, arr, (size_t)n * sizeof(float));
    ce_ring_push(ring, &m);
}

/* ---- Lifecycle ---- */

ce_collector_t* ce_collector_create(ce_ring_buffer_t* ring) {
    if (!ring) return NULL;
    ce_collector_t* c = calloc(1, sizeof(ce_collector_t));
    if (!c) return NULL;
    c->ring = ring;
    return c;
}

void ce_collector_destroy(ce_collector_t* c) {
    free(c);
}

/* ---- Epoch ---- */

void ce_collector_start_epoch(ce_collector_t* c, int epoch) {
    if (!c) return;
    c->current_epoch = epoch;
    c->nan_count = 0;
    c->inf_count = 0;
    c->samples_processed = 0;
    clock_gettime(CLOCK_MONOTONIC, &c->epoch_start);
}

void ce_collector_end_epoch(ce_collector_t* c) {
    if (!c) return;
    float ms = elapsed_ms(&c->epoch_start);
    emit_f32(c->ring, CE_METRIC_EPOCH_TIME_MS,
             (uint32_t)c->current_epoch, ms);
}

/* ---- Phase timers ---- */

void ce_collector_start_forward(ce_collector_t* c) {
    if (c) clock_gettime(CLOCK_MONOTONIC, &c->forward_start);
}

void ce_collector_end_forward(ce_collector_t* c) {
    if (!c) return;
    float ms = elapsed_ms(&c->forward_start);
    emit_f32(c->ring, CE_METRIC_FORWARD_TIME_MS,
             (uint32_t)c->current_epoch, ms);
}

void ce_collector_start_backward(ce_collector_t* c) {
    if (c) clock_gettime(CLOCK_MONOTONIC, &c->backward_start);
}

void ce_collector_end_backward(ce_collector_t* c) {
    if (!c) return;
    float ms = elapsed_ms(&c->backward_start);
    emit_f32(c->ring, CE_METRIC_BACKWARD_TIME_MS,
             (uint32_t)c->current_epoch, ms);
}

void ce_collector_start_optimizer(ce_collector_t* c) {
    if (c) clock_gettime(CLOCK_MONOTONIC, &c->optimizer_start);
}

void ce_collector_end_optimizer(ce_collector_t* c) {
    if (!c) return;
    float ms = elapsed_ms(&c->optimizer_start);
    emit_f32(c->ring, CE_METRIC_OPTIMIZER_TIME_MS,
             (uint32_t)c->current_epoch, ms);
}

void ce_collector_start_data_gen(ce_collector_t* c) {
    if (c) clock_gettime(CLOCK_MONOTONIC, &c->data_gen_start);
}

void ce_collector_end_data_gen(ce_collector_t* c) {
    if (!c) return;
    float ms = elapsed_ms(&c->data_gen_start);
    emit_f32(c->ring, CE_METRIC_DATA_GEN_TIME_MS,
             (uint32_t)c->current_epoch, ms);
}

/* ---- Record metrics ---- */

void ce_collector_record_loss(ce_collector_t* c,
                              float total, float result,
                              float carry, float balance, float z) {
    if (!c) return;
    uint32_t ep = (uint32_t)c->current_epoch;
    emit_f32(c->ring, CE_METRIC_LOSS_TOTAL,   ep, total);
    emit_f32(c->ring, CE_METRIC_LOSS_RESULT,  ep, result);
    emit_f32(c->ring, CE_METRIC_LOSS_CARRY,   ep, carry);
    emit_f32(c->ring, CE_METRIC_LOSS_BALANCE, ep, balance);
    emit_f32(c->ring, CE_METRIC_LOSS_Z,       ep, z);
}

void ce_collector_record_lr(ce_collector_t* c, float lr, float temp) {
    if (!c) return;
    uint32_t ep = (uint32_t)c->current_epoch;
    emit_f32(c->ring, CE_METRIC_LR,          ep, lr);
    emit_f32(c->ring, CE_METRIC_TEMPERATURE, ep, temp);
}

void ce_collector_record_memory(ce_collector_t* c,
                                const ce_memory_stats_t* stats) {
    if (!c || !stats) return;
    uint32_t ep = (uint32_t)c->current_epoch;
    float cur_mb = (float)stats->current_bytes / (1024.0f * 1024.0f);
    float peak_mb = (float)stats->peak_bytes / (1024.0f * 1024.0f);
    emit_f32(c->ring, CE_METRIC_MEM_CURRENT,     ep, cur_mb);
    emit_f32(c->ring, CE_METRIC_MEM_PEAK,        ep, peak_mb);
    emit_f32(c->ring, CE_METRIC_MEM_GROWTH_RATE, ep,
             (float)stats->growth_rate);
}

void ce_collector_record_grad_norm(ce_collector_t* c, float norm) {
    if (!c) return;
    emit_f32(c->ring, CE_METRIC_GRAD_NORM,
             (uint32_t)c->current_epoch, norm);
}

void ce_collector_record_expert_usage(ce_collector_t* c, float usage[4]) {
    if (!c || !usage) return;
    emit_f32_arr(c->ring, CE_METRIC_EXPERT_USAGE,
                 (uint32_t)c->current_epoch, usage, 4);
}

void ce_collector_record_throughput(ce_collector_t* c, float samples_per_sec) {
    if (!c) return;
    emit_f32(c->ring, CE_METRIC_THROUGHPUT,
             (uint32_t)c->current_epoch, samples_per_sec);
}

void ce_collector_record_anomalies(ce_collector_t* c,
                                   int64_t nan_count, int64_t inf_count) {
    if (!c) return;
    c->nan_count += nan_count;
    c->inf_count += inf_count;
    uint32_t ep = (uint32_t)c->current_epoch;
    emit_i64(c->ring, CE_METRIC_NAN_COUNT, ep, c->nan_count);
    emit_i64(c->ring, CE_METRIC_INF_COUNT, ep, c->inf_count);
}

void ce_collector_record_curriculum(ce_collector_t* c, int stage) {
    if (!c) return;
    emit_i64(c->ring, CE_METRIC_CURRICULUM_STAGE,
             (uint32_t)c->current_epoch, (int64_t)stage);
}
