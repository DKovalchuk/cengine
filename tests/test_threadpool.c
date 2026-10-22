/*
 * test_threadpool.c — shutdown/lost-wakeup regression tests (major B3).
 *
 * ce_threadpool_destroy() (and the create-rollback path) set
 * pool->shutdown and broadcast queue->not_empty OUTSIDE queue->lock,
 * while workers read the flag under that lock in queue_pop().  A worker
 * can evaluate the predicate (shutdown == false), then the flag+broadcast
 * happen before it enters pthread_cond_wait — the broadcast has no waiter,
 * the worker sleeps forever, and destroy()'s pthread_join hangs (classic
 * lost wakeup).  The unsynchronised flag write is also a C11 data race.
 *
 * The stress test hammers create/submit/destroy cycles with watchdog
 * alarm(60): on the buggy code the hang surfaces as SIGALRM killing the
 * test binary instead of a silent CI timeout.  The deterministic proof
 * is the ThreadSanitizer run (see PR): flag access must be ordered by
 * queue->lock.
 */
#include "cengine.h"
#include "threadpool.h"
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

static void bump(void* arg) {
    atomic_fetch_add((atomic_int*)arg, 1);
}

/* Variant 1: drain via wait_all, then destroy — workers are parked. */
static int cycle_drain(int n_workers, int n_tasks) {
    ce_threadpool_t* pool = ce_threadpool_create(n_workers);
    if (!pool) return -1;
    atomic_int done;
    atomic_init(&done, 0);
    for (int t = 0; t < n_tasks; t++) {
        if (ce_threadpool_submit(pool, bump, &done) != CE_OK) {
            ce_threadpool_destroy(pool);
            return -1;
        }
    }
    if (ce_threadpool_wait_all(pool) != CE_OK) {
        ce_threadpool_destroy(pool);
        return -1;
    }
    int got = atomic_load(&done);
    ce_threadpool_destroy(pool);          /* ← join must not hang */
    return (got == n_tasks) ? 0 : -1;
}

/* Variant 2: destroy with tasks still queued/in flight — workers are
 * cycling through queue_pop, which is exactly where the lost wakeup
 * window lives. */
static int cycle_drain_abrupt(int n_workers, int n_tasks) {
    ce_threadpool_t* pool = ce_threadpool_create(n_workers);
    if (!pool) return -1;
    atomic_int done;
    atomic_init(&done, 0);
    for (int t = 0; t < n_tasks; t++) {
        if (ce_threadpool_submit(pool, bump, &done) != CE_OK) {
            ce_threadpool_destroy(pool);
            return -1;
        }
    }
    ce_threadpool_destroy(pool);          /* drains, then joins */
    return (atomic_load(&done) == n_tasks) ? 0 : -1;
}

static void test_shutdown_cycles(void) {
    TEST("threadpool: 600 create/submit/destroy cycles don't hang");
    const int workers[] = {1, 2, 4, 8};

    /* Watchdog: a lost wakeup in destroy() hangs the join; convert the
     * hang into a visible signal-death instead of a CI timeout. */
    alarm(60);

    for (int w = 0; w < 4; w++) {
        for (int c = 0; c < 75; c++) {
            int rc = (c % 2) ? cycle_drain(workers[w], 16)
                             : cycle_drain_abrupt(workers[w], 16);
            if (rc != 0) {
                char buf[128];
                snprintf(buf, sizeof buf,
                         "cycle failed (workers=%d, iter=%d)",
                         workers[w], c);
                FAIL(buf);
                return;
            }
        }
    }

    alarm(0);
    PASS();
}

int main(void) {
    printf("\n=== CEngine Threadpool Tests ===\n\n");

    test_shutdown_cycles();

    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
