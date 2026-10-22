#ifndef CENGINE_H
#define CENGINE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version */
#define CENGINE_VERSION_MAJOR 0
#define CENGINE_VERSION_MINOR 1
#define CENGINE_VERSION_PATCH 0

/* Error codes */
typedef enum {
    CE_OK = 0,
    CE_ERR_INVALID_ARG,
    CE_ERR_OUT_OF_MEMORY,
    CE_ERR_INVALID_SHAPE,
    CE_ERR_INVALID_DTYPE,
    CE_ERR_BACKEND_NOT_FOUND,
    CE_ERR_IO,
    CE_ERR_OVERFLOW,
    CE_ERR_THREAD,
    CE_ERR_NOT_INITIALIZED,
    CE_ERR_NOT_IMPLEMENTED,
} ce_status_t;

const char* ce_status_str(ce_status_t status);

/* Initialization / shutdown */
ce_status_t ce_init(void);
void        ce_shutdown(void);

/* Global config */
typedef struct {
    int   num_threads;       /* 0 = auto-detect */
    bool  enable_telemetry;
    bool  enable_perf_counters;
    bool  enable_debug_logging;  /* gate fprintf(stderr, …) diagnostics */
    size_t memory_pool_size; /* 0 = auto */
} ce_config_t;

ce_status_t ce_configure(const ce_config_t* config);
const ce_config_t* ce_get_config(void);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_H */
