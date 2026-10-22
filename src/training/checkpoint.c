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

/* Read a single parameter's header + data into buf, validating the
 * stored metadata against the model's spec.  Data goes to the caller's
 * buffer — never directly into the model (see load_file's commit phase). */
static int read_param(FILE* f, const ce_tensor_t* dest, float* buf)
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

    if (fread(buf, sizeof(float), (size_t)numel, f) != (size_t)numel)
        return -1;

    return 0;
}

/* Parse ONE checkpoint file completely into a scratch block and commit
 * to the model only after every parameter validated.  The old flow
 * overwrote model tensors one by one while parsing, so a truncated or
 * corrupt file left the model holding a mix of old and new weights. */
static ce_status_t load_file(FILE* f, ce_module_t* model)
{
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, CE_CKPT_MAGIC, 4) != 0)
        return CE_ERR_IO;

    uint32_t version;
    if (read_u32(f, &version) || version != CE_CKPT_VERSION)
        return CE_ERR_IO;

    int32_t  epoch;
    float    loss;
    uint32_t n_params;
    if (read_i32(f, &epoch))    return CE_ERR_IO;
    if (read_f32(f, &loss))     return CE_ERR_IO;
    if (read_u32(f, &n_params)) return CE_ERR_IO;

    int model_params = ce_module_param_count(model);
    if ((int)n_params != model_params) {
        fprintf(stderr, "ce_checkpoint_load: checkpoint has %u params, "
                "model has %d\n", n_params, model_params);
        return CE_ERR_INVALID_SHAPE;
    }

    /* One scratch block sized from the model (metadata was validated
     * against it, so the file's payloads fit exactly). */
    int64_t total = 0;
    for (int i = 0; i < model_params; i++) {
        ce_tensor_t* p = ce_module_get_parameter(model, i);
        if (!p) return CE_ERR_NOT_INITIALIZED;
        total += p->numel;
    }
    float* block = NULL;
    if (total > 0) {
        block = (float*)malloc((size_t)total * sizeof(float));
        if (!block) return CE_ERR_OUT_OF_MEMORY;
    }

    int64_t off = 0;
    for (int i = 0; i < model_params; i++) {
        ce_tensor_t* p = ce_module_get_parameter(model, i);
        if (read_param(f, p, block + off)) {
            free(block);
            return CE_ERR_IO;
        }
        off += p->numel;
    }

    /* Commit: everything parsed and validated. */
    off = 0;
    for (int i = 0; i < model_params; i++) {
        ce_tensor_t* p = ce_module_get_parameter(model, i);
        memcpy(p->data, block + off, (size_t)p->numel * sizeof(float));
        off += p->numel;
    }
    free(block);
    return CE_OK;
}

static ce_status_t try_load_file(const char* path, ce_module_t* model)
{
    FILE* f = fopen(path, "rb");
    if (!f) return CE_ERR_IO;
    ce_status_t st = load_file(f, model);
    fclose(f);
    return st;
}

static int cmp_epoch_desc(const void* a, const void* b) {
    return *(const int*)b - *(const int*)a;
}

ce_status_t ce_checkpoint_load(const char* dir, ce_module_t* model)
{
    if (!dir || !dir[0] || !model) return CE_ERR_INVALID_ARG;

    /* Try "latest" first — a dangling or corrupt link falls through to
     * the scan below instead of failing the whole load. */
    char path[512];
    snprintf(path, sizeof(path), "%s/checkpoint_latest.bin", dir);
    if (try_load_file(path, model) == CE_OK) return CE_OK;

    /* Scan the directory and try candidates NEWEST FIRST: the first file
     * that validates completely wins.  The old code committed to the
     * single highest epoch without validating it, so one corrupt file
     * made the directory unloadable. */
    DIR* d = opendir(dir);
    if (!d) return CE_ERR_IO;

    int cap = 32, n_ep = 0;
    int* epochs = (int*)malloc((size_t)cap * sizeof(int));
    if (!epochs) { closedir(d); return CE_ERR_OUT_OF_MEMORY; }

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        int ep;
        if (sscanf(ent->d_name, "checkpoint_epoch_%d.bin", &ep) == 1) {
            if (n_ep == cap) {
                cap *= 2;
                int* grown = (int*)realloc(epochs, (size_t)cap * sizeof(int));
                if (!grown) { free(epochs); closedir(d); return CE_ERR_OUT_OF_MEMORY; }
                epochs = grown;
            }
            epochs[n_ep++] = ep;
        }
    }
    closedir(d);

    qsort(epochs, (size_t)n_ep, sizeof(int), cmp_epoch_desc);

    ce_status_t st = CE_ERR_IO;
    for (int i = 0; i < n_ep; i++) {
        snprintf(path, sizeof(path), "%s/checkpoint_epoch_%d.bin", dir,
                 epochs[i]);
        if (try_load_file(path, model) == CE_OK) { st = CE_OK; break; }
    }
    free(epochs);
    return st;
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

    /* Use symlink if supported, otherwise copy.  The target is the
     * BASENAME: a symlink resolves relative to its own directory, so a
     * full path dangles whenever `dir` itself is relative ("ckpt/ckpt/…"). */
    unlink(latest);
    char link_target[64];
    snprintf(link_target, sizeof(link_target), "checkpoint_epoch_%d.bin",
             epoch);
    if (symlink(link_target, latest) != 0) {
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

