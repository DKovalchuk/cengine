#define _GNU_SOURCE
#include "cengine.h"
#include "telemetry.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <errno.h>

/* perf_event_open wrapper */
static int perf_open(uint32_t type, uint64_t config) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.read_format = 0;

    int fd = (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    return fd;
}

static int perf_open_and_enable(uint32_t type, uint64_t config) {
    int fd = perf_open(type, config);
    if (fd >= 0) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }
    return fd;
}

static uint64_t perf_read_fd(int fd) {
    if (fd < 0) return 0;
    uint64_t count = 0;
    ssize_t n = read(fd, &count, sizeof(count));
    if (n != sizeof(count)) return 0;
    return count;
}

ce_perf_t* ce_perf_create(void) {
    ce_perf_t* p = calloc(1, sizeof(ce_perf_t));
    if (!p) return NULL;

    p->fd_instructions = perf_open_and_enable(
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    p->fd_cycles = perf_open_and_enable(
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    p->fd_cache_references = perf_open_and_enable(
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    p->fd_cache_misses = perf_open_and_enable(
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    p->fd_branch_instructions = perf_open_and_enable(
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    p->fd_branch_misses = perf_open_and_enable(
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);

    /* If any critical fd failed, mark unavailable */
    if (p->fd_instructions < 0 || p->fd_cycles < 0) {
        p->available = false;
    } else {
        p->available = true;
    }

    return p;
}

void ce_perf_destroy(ce_perf_t* p) {
    if (!p) return;
    if (p->fd_instructions >= 0)      close(p->fd_instructions);
    if (p->fd_cycles >= 0)            close(p->fd_cycles);
    if (p->fd_cache_references >= 0)  close(p->fd_cache_references);
    if (p->fd_cache_misses >= 0)      close(p->fd_cache_misses);
    if (p->fd_branch_instructions >= 0) close(p->fd_branch_instructions);
    if (p->fd_branch_misses >= 0)     close(p->fd_branch_misses);
    free(p);
}

void ce_perf_read(ce_perf_t* p, float* ipc,
                  float* l2_miss_rate, float* llc_miss_rate,
                  float* branch_miss_rate) {
    if (!p) return;

    /* Defaults */
    if (ipc) *ipc = 0.0f;
    if (l2_miss_rate) *l2_miss_rate = 0.0f;
    if (llc_miss_rate) *llc_miss_rate = 0.0f;
    if (branch_miss_rate) *branch_miss_rate = 0.0f;

    if (!p->available) return;

    uint64_t instructions = perf_read_fd(p->fd_instructions);
    uint64_t cycles       = perf_read_fd(p->fd_cycles);
    uint64_t cache_ref    = perf_read_fd(p->fd_cache_references);
    uint64_t cache_miss   = perf_read_fd(p->fd_cache_misses);
    uint64_t branch_instr = perf_read_fd(p->fd_branch_instructions);
    uint64_t branch_miss  = perf_read_fd(p->fd_branch_misses);

    /* IPC = instructions / cycles */
    if (ipc && cycles > 0) {
        *ipc = (float)instructions / (float)cycles;
    }

    /* L2 miss rate = cache_misses / cache_references
     * (PERF_COUNT_HW_CACHE_MISSES maps to L2 on most x86) */
    if (l2_miss_rate && cache_ref > 0) {
        *l2_miss_rate = (float)cache_miss / (float)cache_ref;
    }

    /* LLC miss rate: same metric on many systems (kernel maps it) */
    if (llc_miss_rate && cache_ref > 0) {
        *llc_miss_rate = (float)cache_miss / (float)cache_ref;
    }

    /* Branch miss rate = branch_misses / branch_instructions */
    if (branch_miss_rate && branch_instr > 0) {
        *branch_miss_rate = (float)branch_miss / (float)branch_instr;
    }

    /* Reset counters for next interval */
    if (p->fd_instructions >= 0)      ioctl(p->fd_instructions, PERF_EVENT_IOC_RESET, 0);
    if (p->fd_cycles >= 0)            ioctl(p->fd_cycles, PERF_EVENT_IOC_RESET, 0);
    if (p->fd_cache_references >= 0)  ioctl(p->fd_cache_references, PERF_EVENT_IOC_RESET, 0);
    if (p->fd_cache_misses >= 0)      ioctl(p->fd_cache_misses, PERF_EVENT_IOC_RESET, 0);
    if (p->fd_branch_instructions >= 0) ioctl(p->fd_branch_instructions, PERF_EVENT_IOC_RESET, 0);
    if (p->fd_branch_misses >= 0)     ioctl(p->fd_branch_misses, PERF_EVENT_IOC_RESET, 0);
}
