#include "optimizer.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ce_scheduler_t* ce_scheduler_create(float max_lr, float pct_start,
                                     float div_factor, float final_div_factor,
                                     int total_steps) {
    if (total_steps <= 0 || div_factor <= 0.0f || final_div_factor <= 0.0f) return NULL;
    if (pct_start <= 0.0f || pct_start >= 1.0f) return NULL;

    ce_scheduler_t* sched = (ce_scheduler_t*)malloc(sizeof(ce_scheduler_t));
    if (!sched) return NULL;

    sched->max_lr = max_lr;
    sched->min_lr = max_lr / final_div_factor;
    sched->pct_start = pct_start;
    sched->div_factor = div_factor;
    sched->final_div_factor = final_div_factor;
    sched->total_steps = total_steps;
    sched->current_step = 0;
    sched->current_lr = max_lr / div_factor;  /* initial lr */

    return sched;
}

float ce_scheduler_step(ce_scheduler_t* sched) {
    if (!sched) return 0.0f;

    sched->current_step++;
    if (sched->current_step > sched->total_steps) {
        sched->current_step = sched->total_steps;
    }

    int warmup_steps = (int)(sched->total_steps * sched->pct_start);
    int anneal_steps = sched->total_steps - warmup_steps;

    if (sched->current_step <= warmup_steps) {
        /* Linear warmup: initial_lr -> max_lr */
        float warmup_progress = (float)sched->current_step / (float)warmup_steps;
        float initial_lr = sched->max_lr / sched->div_factor;
        sched->current_lr = initial_lr + (sched->max_lr - initial_lr) * warmup_progress;
    } else {
        /* Cosine annealing: max_lr -> min_lr */
        float anneal_progress = (float)(sched->current_step - warmup_steps) / (float)anneal_steps;
        float cos_factor = 0.5f * (1.0f + cosf((float)M_PI * anneal_progress));
        sched->current_lr = sched->min_lr + (sched->max_lr - sched->min_lr) * cos_factor;
    }

    return sched->current_lr;
}

void ce_scheduler_reset(ce_scheduler_t* sched) {
    if (!sched) return;
    sched->current_step = 0;
    sched->current_lr = sched->max_lr / sched->div_factor;
}

void ce_scheduler_destroy(ce_scheduler_t* sched) {
    if (sched) free(sched);
}
