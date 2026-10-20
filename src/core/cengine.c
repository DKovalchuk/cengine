#include "cengine.h"
#include "memory.h"
#include "compute.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static ce_config_t g_config = {0};
static bool g_initialized = false;

const char* ce_status_str(ce_status_t status) {
    switch (status) {
        case CE_OK:                 return "OK";
        case CE_ERR_INVALID_ARG:    return "Invalid argument";
        case CE_ERR_OUT_OF_MEMORY:  return "Out of memory";
        case CE_ERR_INVALID_SHAPE:  return "Invalid shape";
        case CE_ERR_INVALID_DTYPE:  return "Invalid dtype";
        case CE_ERR_BACKEND_NOT_FOUND: return "Backend not found";
        case CE_ERR_IO:             return "I/O error";
        case CE_ERR_OVERFLOW:       return "Overflow";
        case CE_ERR_THREAD:         return "Thread error";
        case CE_ERR_NOT_INITIALIZED:return "Not initialized";
        default:                    return "Unknown error";
    }
}

ce_status_t ce_init(void) {
    if (g_initialized) return CE_OK;

    ce_status_t st = ce_tracker_init();
    if (st != CE_OK) return st;

    g_initialized = true;
    return CE_OK;
}

void ce_shutdown(void) {
    if (!g_initialized) return;
    ce_tracker_shutdown();
    g_initialized = false;
}

ce_status_t ce_configure(const ce_config_t* config) {
    if (!config) return CE_ERR_INVALID_ARG;
    g_config = *config;

    if (g_config.num_threads <= 0) {
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        g_config.num_threads = (nprocs > 0) ? (int)nprocs : 4;
    }

    if (g_config.enable_debug_logging) {
        fprintf(stderr, "[CEngine] Debug logging enabled (threads=%d)\n",
                g_config.num_threads);
    }

    return CE_OK;
}

const ce_config_t* ce_get_config(void) {
    return &g_config;
}
