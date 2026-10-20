#include "compute.h"
#include <string.h>
#include <stdio.h>

#define MAX_BACKENDS 4

static ce_compute_backend_t* g_backends[MAX_BACKENDS];
static int g_backend_count = 0;
static ce_compute_backend_t* g_default = NULL;

/* Forward declaration of the CPU backend (implemented in math/) */
extern ce_compute_backend_t* ce_cpu_backend_create(void);

ce_status_t ce_register_backend(ce_compute_backend_t* backend) {
    if (!backend || g_backend_count >= MAX_BACKENDS) return CE_ERR_INVALID_ARG;
    g_backends[g_backend_count++] = backend;
    if (!g_default || backend->device == CE_DEVICE_CPU) {
        g_default = backend;
    }
    return CE_OK;
}

ce_compute_backend_t* ce_get_backend(ce_device_t device) {
    for (int i = 0; i < g_backend_count; i++) {
        if (g_backends[i]->device == device) return g_backends[i];
    }
    return NULL;
}

ce_compute_backend_t* ce_get_default_backend(void) {
    if (!g_default) {
        /* Lazy-init CPU backend */
        g_default = ce_cpu_backend_create();
        if (g_default) {
            g_backends[g_backend_count++] = g_default;
        }
    }
    return g_default;
}
