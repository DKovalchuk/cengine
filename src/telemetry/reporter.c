#include "cengine.h"
#include "telemetry.h"

#include <stdlib.h>
#include <string.h>

ce_reporter_t* ce_reporter_create(ce_ring_buffer_t* ring) {
    if (!ring) return NULL;
    ce_reporter_t* r = calloc(1, sizeof(ce_reporter_t));
    if (!r) return NULL;
    r->ring = ring;
    return r;
}

void ce_reporter_destroy(ce_reporter_t* r) {
    free(r);
}

/* Apply a single metric to the snapshot */
static void apply_metric(ce_telemetry_snapshot_t* snap, const ce_metric_t* m) {
    snap->last_update_ns = m->timestamp_ns;
    snap->current_epoch = (int)m->epoch;

    switch (m->metric_type) {
    case CE_METRIC_LOSS_TOTAL:       snap->loss_total = m->value.f32; break;
    case CE_METRIC_LOSS_RESULT:      snap->loss_result = m->value.f32; break;
    case CE_METRIC_LOSS_CARRY:       snap->loss_carry = m->value.f32; break;
    case CE_METRIC_LOSS_BALANCE:     snap->loss_balance = m->value.f32; break;
    case CE_METRIC_LOSS_Z:           snap->loss_z = m->value.f32; break;
    case CE_METRIC_LR:               snap->lr = m->value.f32; break;
    case CE_METRIC_TEMPERATURE:      snap->temperature = m->value.f32; break;
    case CE_METRIC_MEM_CURRENT:      snap->mem_current_mb = m->value.f32; break;
    case CE_METRIC_MEM_PEAK:         snap->mem_peak_mb = m->value.f32; break;
    case CE_METRIC_MEM_GROWTH_RATE:  snap->mem_growth_rate = m->value.f32; break;
    case CE_METRIC_GRAD_NORM:        snap->grad_norm = m->value.f32; break;
    case CE_METRIC_EXPERT_USAGE:
        memcpy(snap->expert_usage, m->value.f32_arr, 4 * sizeof(float));
        break;
    case CE_METRIC_THROUGHPUT:       snap->throughput = m->value.f32; break;
    case CE_METRIC_EPOCH_TIME_MS:    snap->epoch_time_ms = m->value.f32; break;
    case CE_METRIC_FORWARD_TIME_MS:  snap->forward_time_ms = m->value.f32; break;
    case CE_METRIC_BACKWARD_TIME_MS: snap->backward_time_ms = m->value.f32; break;
    case CE_METRIC_OPTIMIZER_TIME_MS:snap->optimizer_time_ms = m->value.f32; break;
    case CE_METRIC_DATA_GEN_TIME_MS: snap->data_gen_time_ms = m->value.f32; break;
    case CE_METRIC_CPU_UTIL:         snap->cpu_util = m->value.f32; break;
    case CE_METRIC_HW_IPC:           snap->hw_ipc = m->value.f32; break;
    case CE_METRIC_HW_L2_MISS_RATE:  snap->hw_l2_miss_rate = m->value.f32; break;
    case CE_METRIC_HW_LLC_MISS_RATE: snap->hw_llc_miss_rate = m->value.f32; break;
    case CE_METRIC_HW_BRANCH_MISS_RATE:
        snap->hw_branch_miss_rate = m->value.f32; break;
    case CE_METRIC_ACTIVE_WORKERS:
        snap->active_workers = (int)m->value.i64; break;
    case CE_METRIC_THREADPOOL_QUEUE_DEPTH:
        snap->threadpool_queue_depth = (int)m->value.i64; break;
    case CE_METRIC_NAN_COUNT:        snap->nan_count = m->value.i64; break;
    case CE_METRIC_INF_COUNT:        snap->inf_count = m->value.i64; break;
    case CE_METRIC_CURRICULUM_STAGE:
        snap->curriculum_stage = (int)m->value.i64; break;
    default: break;
    }
}

int ce_reporter_poll(ce_reporter_t* r) {
    if (!r) return 0;

    /* Drain up to 256 metrics per poll to avoid blocking too long */
    ce_metric_t batch[256];
    int total = 0;

    for (;;) {
        int n = ce_ring_batch_pop(r->ring, batch, 256);
        if (n <= 0) break;

        for (int i = 0; i < n; i++) {
            if (batch[i].flags & CE_METRIC_FLAG_VALID) {
                apply_metric(&r->snapshot, &batch[i]);
            }
        }
        total += n;

        if (n < 256) break;  /* drained */
    }

    return total;
}

ce_telemetry_snapshot_t ce_reporter_get_snapshot(const ce_reporter_t* r) {
    if (!r) {
        ce_telemetry_snapshot_t empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }
    return r->snapshot;
}
