/*
 * checkpoint.c — Binary checkpoint save/load for model weights.
 *
 * Format (v1):
 *   Header:
 *     char    magic[4]   = "CECK"
 *     uint32  version    = 1
 *     int32   epoch
 *     float   loss
 *     uint32  n_params
 *   Per parameter (× n_params):
 *     uint32  name_len
 *     char    name[name_len]
 *     int32   ndim
 *     int64   shape[ndim]
 *     int64   numel
 *     float   data[numel]
 */
#include "trainer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>

#define CE_CKPT_MAGIC   "CECK"
#define CE_CKPT_VERSION 1

/* ---- Helpers ---- */

static int write_u32(FILE* f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1 ? 0 : -1; }
static int write_i32(FILE* f, int32_t  v) { return fwrite(&v, 4, 1, f) == 1 ? 0 : -1; }
static int write_f32(FILE* f, float    v) { return fwrite(&v, 4, 1, f) == 1 ? 0 : -1; }
static int write_i64(FILE* f, int64_t  v) { return fwrite(&v, 8, 1, f) == 1 ? 0 : -1; }

static int read_u32(FILE* f, uint32_t* v) { return fread(v, 4, 1, f) == 1 ? 0 : -1; }
static int read_i32(FILE* f, int32_t*  v) { return fread(v, 4, 1, f) == 1 ? 0 : -1; }
static int read_f32(FILE* f, float*    v) { return fread(v, 4, 1, f) == 1 ? 0 : -1; }
static int read_i64(FILE* f, int64_t*  v) { return fread(v, 8, 1, f) == 1 ? 0 : -1; }

/* Ensure directory exists (best-effort). */
static void ensure_dir(const char* path)
{
    struct stat st;
    if (stat(path, &st) == 0) return;
    mkdir(path, 0755);
}

/* Write a single parameter tensor to file. */
static int write_param(FILE* f, const ce_tensor_t* t)
{
    uint32_t name_len = (uint32_t)strlen(t->name);
    if (write_u32(f, name_len))               return -1;
    if (fwrite(t->name, 1, name_len, f) != name_len) return -1;
    if (write_i32(f, (int32_t)t->ndim))      return -1;
    for (int d = 0; d < t->ndim; d++) {
        if (write_i64(f, t->shape[d]))        return -1;
    }
    if (write_i64(f, t->numel))               return -1;
    if (fwrite(t->data, sizeof(float), (size_t)t->numel, f) != (size_t)t->numel)
        return -1;
    return 0;
}

/* Read a single parameter tensor from file into a pre-existing model parameter. */
static int read_param(FILE* f, ce_tensor_t* dest)
{
    uint32_t name_len;
    if (read_u32(f, &name_len)) return -1;
    if (name_len >= sizeof(dest->name)) return -1;

    char name[64] = {0};
    if (fread(name, 1, name_len, f) != name_len) return -1;

    int32_t ndim;
    if (read_i32(f, &ndim)) return -1;
    if (ndim != dest->ndim) return -1;

    for (int d = 0; d < ndim; d++) {
        int64_t dim;
        if (read_i64(f, &dim)) return -1;
        if (dim != dest->shape[d]) return -1;
    }

    int64_t numel;
    if (read_i64(f, &numel)) return -1;
    if (numel != dest->numel) return -1;

    if (fread(dest->data, sizeof(float), (size_t)numel, f) != (size_t)numel)
        return -1;

    return 0;
}

/* ====================================================================
 * Public API
 * ==================================================================== */

ce_status_t ce_checkpoint_save(const char* dir, int epoch, float loss,
                                ce_module_t* model, void* optimizer_state)
{
    (void)optimizer_state; /* reserved for future use */

    if (!dir || !dir[0] || !model) return CE_ERR_INVALID_ARG;

    ensure_dir(dir);

    /* Write to epoch-specific file */
    char path[512];
    snprintf(path, sizeof(path), "%s/checkpoint_epoch_%d.bin", dir, epoch);

    /* Atomic save: write into a hidden temp file in the SAME directory
     * and rename() it into place.  Writing the final path directly let a
     * failed or interrupted write leave a truncated file there — "latest"
     * and the highest-epoch scan would then load garbage. */
    char tmp[560];
    snprintf(tmp, sizeof(tmp), "%s/.checkpoint_epoch_%d.tmp", dir, epoch);

    FILE* f = fopen(tmp, "wb");
    if (!f) return CE_ERR_IO;

    int n_params = ce_module_param_count(model);

    /* Header */
    if (fwrite(CE_CKPT_MAGIC, 1, 4, f) != 4)                    goto write_fail;
    if (write_u32(f, CE_CKPT_VERSION))                           goto write_fail;
    if (write_i32(f, (int32_t)epoch))                            goto write_fail;
    if (write_f32(f, loss))                                       goto write_fail;
    if (write_u32(f, (uint32_t)n_params))                        goto write_fail;

    /* Parameters */
    for (int i = 0; i < n_params; i++) {
        ce_tensor_t* p = ce_module_get_parameter(model, i);
        if (!p) goto write_fail;
        if (write_param(f, p)) goto write_fail;
    }

    /* Surface deferred stdio errors before committing the rename. */
    int flush_rc = fflush(f);
    int close_rc = fclose(f);
    f = NULL;
    if (flush_rc != 0 || close_rc != 0) {
        unlink(tmp);
        return CE_ERR_IO;
    }
    if (rename(tmp, path) != 0) {
        unlink(tmp);
        return CE_ERR_IO;
    }

    /* Also write to "latest" for easy loading */
    char latest[512];
    snprintf(latest, sizeof(latest), "%s/checkpoint_latest.bin", dir);

    /* Use symlink if supported, otherwise copy */
    unlink(latest);
    if (symlink(path, latest) != 0) {
        /* Fallback: byte copy */
        FILE* src = fopen(path, "rb");
        FILE* dst = fopen(latest, "wb");
        if (src && dst) {
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), src)) > 0) {
                fwrite(buf, 1, n, dst);
            }
        }
        if (src) fclose(src);
        if (dst) fclose(dst);
    }

    return CE_OK;

write_fail:
    if (f) fclose(f);
    unlink(tmp);
    return CE_ERR_IO;
}

ce_status_t ce_checkpoint_load(const char* dir, ce_module_t* model)
{
    if (!dir || !dir[0] || !model) return CE_ERR_INVALID_ARG;

    /* Try "latest" first, then scan for highest epoch */
    char path[512];
    snprintf(path, sizeof(path), "%s/checkpoint_latest.bin", dir);

    FILE* f = fopen(path, "rb");
    if (!f) {
        /* Scan directory for the highest-numbered checkpoint */
        DIR* d = opendir(dir);
        if (!d) return CE_ERR_IO;

        int best_epoch = -1;
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            int ep;
            if (sscanf(ent->d_name, "checkpoint_epoch_%d.bin", &ep) == 1) {
                if (ep > best_epoch) best_epoch = ep;
            }
        }
        closedir(d);

        if (best_epoch < 0) return CE_ERR_IO;

        snprintf(path, sizeof(path), "%s/checkpoint_epoch_%d.bin", dir, best_epoch);
        f = fopen(path, "rb");
        if (!f) return CE_ERR_IO;
    }

    /* Read and verify header */
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, CE_CKPT_MAGIC, 4) != 0) {
        fclose(f);
        return CE_ERR_IO;
    }

    uint32_t version;
    if (read_u32(f, &version) || version != CE_CKPT_VERSION) {
        fclose(f);
        return CE_ERR_IO;
    }

    int32_t  epoch;
    float    loss;
    uint32_t n_params;
    if (read_i32(f, &epoch))    { fclose(f); return CE_ERR_IO; }
    if (read_f32(f, &loss))     { fclose(f); return CE_ERR_IO; }
    if (read_u32(f, &n_params)) { fclose(f); return CE_ERR_IO; }

    /* Verify parameter count matches */
    int model_params = ce_module_param_count(model);
    if ((int)n_params != model_params) {
        fprintf(stderr, "ce_checkpoint_load: checkpoint has %u params, "
                "model has %d\n", n_params, model_params);
        fclose(f);
        return CE_ERR_INVALID_SHAPE;
    }

    /* Load each parameter */
    for (int i = 0; i < (int)n_params; i++) {
        ce_tensor_t* p = ce_module_get_parameter(model, i);
        if (!p) { fclose(f); return CE_ERR_NOT_INITIALIZED; }
        if (read_param(f, p)) {
            fclose(f);
            return CE_ERR_IO;
        }
    }

    fclose(f);
    return CE_OK;
}
