/*
 * test_checkpoint.c — checkpoint save/load reliability tests (major B1/B2).
 *
 * B1: ce_checkpoint_save wrote directly to the final path — a failed or
 * interrupted write left a truncated file at `checkpoint_epoch_N.bin`
 * (and "latest" kept pointing at it), so the next load failed or read
 * garbage.  The save must be atomic: write a hidden temp file in the
 * same directory, flush, then rename().
 *
 * B2: the "latest" symlink was created with the FULL path as target
 * (dangling whenever `dir` is relative — the target resolves relative
 * to the link's own directory); the load scan picked the highest epoch
 * without validating it; and read_param overwrote model tensors as it
 * parsed, so a truncated file left the model with MIXED weights from
 * two sources (no rollback).
 *
 * Failure injection for B1 uses RLIMIT_FSIZE + ignored SIGXFSZ, which
 * makes a mid-write failure deterministic on Linux.
 */
#include "cengine.h"
#include "layers.h"
#include "tensor.h"
#include "trainer.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  TEST %-52s ", name); \
    tests_run++; \
} while(0)

#define PASS() do { tests_passed++; printf("[PASS]\n"); } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); } while(0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

/* Deterministic fill in [-0.5, 0.5]. */
static void lcg_fill(float* p, int64_t n, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
    }
}

static int values_equal(const ce_tensor_t* a, const ce_tensor_t* b) {
    if (a->numel != b->numel) return 0;
    for (int64_t i = 0; i < a->numel; i++)
        if (a->data[i] != b->data[i]) return 0;
    return 1;
}

/* Module with two sized parameters, filled from the given seed. */
static ce_module_t* make_model(uint32_t seed) {
    ce_module_t* m = ce_module_create("ckpt_model");
    if (!m) return NULL;
    int64_t s0[2] = {64, 32};
    int64_t s1[1] = {128};
    ce_tensor_t* p0 = ce_tensor_zeros(s0, 2);
    ce_tensor_t* p1 = ce_tensor_zeros(s1, 1);
    ce_tensor_t* g0 = ce_tensor_zeros(s0, 2);
    ce_tensor_t* g1 = ce_tensor_zeros(s1, 1);
    if (!p0 || !p1 || !g0 || !g1) return NULL;
    strncpy(p0->name, "w0", sizeof p0->name - 1);
    strncpy(p1->name, "w1", sizeof p1->name - 1);
    lcg_fill(p0->data, p0->numel, seed);
    lcg_fill(p1->data, p1->numel, seed + 1);
    ce_module_add_parameter(m, p0, g0);
    ce_module_add_parameter(m, p1, g1);
    return m;
}

static int dir_contains(const char* dir, const char* suffix) {
    DIR* d = opendir(dir);
    if (!d) return 0;
    struct dirent* ent;
    int found = 0;
    while ((ent = readdir(d)) != NULL) {
        const char* nm = ent->d_name;
        size_t nl = strlen(nm), sl = strlen(suffix);
        if (nl >= sl && strcmp(nm + nl - sl, suffix) == 0) { found = 1; break; }
    }
    closedir(d);
    return found;
}

/* =====================================================================
 * B1 — atomic save
 * ===================================================================== */

static void test_save_load_roundtrip(void) {
    TEST("checkpoint: save/load roundtrip preserves values");
    char dir[] = "/tmp/cengine_ckpt_XXXXXX";
    ASSERT(mkdtemp(dir) != NULL, "mkdtemp failed");

    ce_module_t* m = make_model(42);
    ASSERT(m != NULL, "model create");

    ASSERT(ce_checkpoint_save(dir, 3, 0.25f, m, NULL) == CE_OK,
           "save failed");

    ce_module_t* m2 = make_model(0);   /* different values */
    ASSERT(m2 != NULL, "model2 create");
    ASSERT(ce_checkpoint_load(dir, m2) == CE_OK, "load failed");

    ce_tensor_t* w0 = ce_module_get_parameter(m2, 0);
    ce_tensor_t* w1 = ce_module_get_parameter(m2, 1);
    ce_tensor_t* r0 = ce_module_get_parameter(m, 0);
    ce_tensor_t* r1 = ce_module_get_parameter(m, 1);
    ASSERT(values_equal(w0, r0) && values_equal(w1, r1),
           "loaded values differ from saved");

    ce_module_destroy(m);
    ce_module_destroy(m2);
    PASS();
}

/* Overwriting an existing checkpoint must be atomic: if the rewrite
 * fails mid-write, the previous good file must survive intact and
 * loadable. */
static void test_failed_overwrite_keeps_previous(void) {
    TEST("checkpoint: failed overwrite keeps previous (atomic save)");
    char dir[] = "/tmp/cengine_ckpt_XXXXXX";
    ASSERT(mkdtemp(dir) != NULL, "mkdtemp failed");

    ce_module_t* m = make_model(7);
    ASSERT(m != NULL, "model create");
    ce_tensor_t* orig0 = ce_module_get_parameter(m, 0);
    ce_tensor_t* orig1 = ce_module_get_parameter(m, 1);

    /* Good save first. */
    ASSERT(ce_checkpoint_save(dir, 3, 0.5f, m, NULL) == CE_OK,
           "initial save failed");

    /* Force mid-write failure on the overwrite: payload (≈8.5 KB) far
     * exceeds the file-size limit. */
    struct rlimit old_lim;
    ASSERT(getrlimit(RLIMIT_FSIZE, &old_lim) == 0, "getrlimit failed");
    struct rlimit tight = old_lim;
    tight.rlim_cur = 512;
    void (*old_handler)(int) = signal(SIGXFSZ, SIG_IGN);
    ASSERT(setrlimit(RLIMIT_FSIZE, &tight) == 0, "setrlimit failed");

    ce_status_t st = ce_checkpoint_save(dir, 3, 9.9f, m, NULL);

    setrlimit(RLIMIT_FSIZE, &old_lim);
    signal(SIGXFSZ, old_handler);

    /* Whatever the save reported, the previous checkpoint must still be
     * complete and loadable with its ORIGINAL values — never truncated,
     * never mixed. */
    ce_module_t* m2 = make_model(0);
    ASSERT(m2 != NULL, "model2 create");
    ce_status_t lst = ce_checkpoint_load(dir, m2);
    if (lst != CE_OK) {
        char buf[160];
        snprintf(buf, sizeof buf,
                 "load after failed save returned 0x%x — previous "
                 "checkpoint was clobbered (st=0x%x)", lst, st);
        FAIL(buf);
        ce_module_destroy(m2);
        return;
    }
    ce_tensor_t* w0 = ce_module_get_parameter(m2, 0);
    ce_tensor_t* w1 = ce_module_get_parameter(m2, 1);
    ASSERT(values_equal(w0, orig0) && values_equal(w1, orig1),
           "values after failed save are not the previous checkpoint's");

    /* No temp residue may be left behind. */
    ASSERT(!dir_contains(dir, ".tmp"), "temp file left in checkpoint dir");

    ce_module_destroy(m);
    ce_module_destroy(m2);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Checkpoint Tests ===\n\n");
    ce_init();

    test_save_load_roundtrip();
    test_failed_overwrite_keeps_previous();

    ce_shutdown();
    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
