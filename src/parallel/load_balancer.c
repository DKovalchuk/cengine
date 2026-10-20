#include "cengine.h"
#include "threadpool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

ce_load_balancer_t* ce_load_balancer_create(int min_workers, int max_workers,
                                            float target_cpu_util) {
    if (min_workers <= 0) min_workers = 1;
    if (max_workers < min_workers) max_workers = min_workers;
    if (target_cpu_util <= 0.0f || target_cpu_util > 1.0f)
        target_cpu_util = 0.75f;

    ce_load_balancer_t* lb = calloc(1, sizeof(ce_load_balancer_t));
    if (!lb) return NULL;

    lb->min_workers = min_workers;
    lb->max_workers = max_workers;
    lb->target_cpu_util = target_cpu_util;
    lb->current_workers = min_workers;
    lb->prev_idle = 0;
    lb->prev_total = 0;

    return lb;
}

/* Read aggregate CPU jiffies from /proc/stat.
 * Returns 0 on success, -1 on error.
 * Fields: user nice system idle iowait irq softirq steal */
static int read_cpu_jiffies(uint64_t* idle_out, uint64_t* total_out) {
    FILE* f = fopen("/proc/stat", "r");
    if (!f) return -1;

    char line[512];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }
    fclose(f);

    /* First line starts with "cpu " (aggregate) */
    if (strncmp(line, "cpu ", 4) != 0) return -1;

    uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
    int n = sscanf(line + 4, "%lu %lu %lu %lu %lu %lu %lu %lu",
                   &user, &nice, &system, &idle, &iowait,
                   &irq, &softirq, &steal);
    if (n < 5) return -1;

    /* Total = sum of all fields */
    uint64_t total = user + nice + system + idle + iowait + irq + softirq;
    if (n >= 8) total += steal;

    *idle_out = idle + iowait;  /* idle includes iowait in Linux */
    *total_out = total;
    return 0;
}

int ce_load_balancer_adjust(ce_load_balancer_t* lb) {
    if (!lb) return -1;

    uint64_t idle = 0, total = 0;
    if (read_cpu_jiffies(&idle, &total) != 0) {
        return lb->current_workers;  /* can't read, keep current */
    }

    /* Need two readings to compute delta */
    if (lb->prev_total == 0) {
        lb->prev_idle = idle;
        lb->prev_total = total;
        return lb->current_workers;
    }

    uint64_t delta_idle  = idle  - lb->prev_idle;
    uint64_t delta_total = total - lb->prev_total;

    lb->prev_idle  = idle;
    lb->prev_total = total;

    if (delta_total == 0) return lb->current_workers;

    float cpu_util = 1.0f - (float)delta_idle / (float)delta_total;

    /* Adjust worker count based on utilization vs target */
    if (cpu_util > lb->target_cpu_util + 0.10f) {
        /* Over-utilized: reduce workers */
        if (lb->current_workers > lb->min_workers) {
            lb->current_workers--;
        }
    } else if (cpu_util < lb->target_cpu_util - 0.10f) {
        /* Under-utilized: increase workers */
        if (lb->current_workers < lb->max_workers) {
            lb->current_workers++;
        }
    }

    return lb->current_workers;
}

void ce_load_balancer_destroy(ce_load_balancer_t* lb) {
    free(lb);
}
