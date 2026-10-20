#define _GNU_SOURCE
#include "cengine.h"
#include "threadpool.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

/* ---- Internal task queue (bounded, mutex-guarded) ---- */

#define TASK_QUEUE_CAP 4096

typedef struct {
    ce_task_t items[TASK_QUEUE_CAP];
    int       head;
    int       tail;
    int       count;
    pthread_mutex_t lock;
    pthread_cond_t  not_empty;
    pthread_cond_t  not_full;
} task_queue_t;

static ce_status_t queue_init(task_queue_t* q) {
    memset(q, 0, sizeof(*q));
    int rc;
    rc = pthread_mutex_init(&q->lock, NULL);
    if (rc) return CE_ERR_THREAD;
    rc = pthread_cond_init(&q->not_empty, NULL);
    if (rc) return CE_ERR_THREAD;
    rc = pthread_cond_init(&q->not_full, NULL);
    if (rc) return CE_ERR_THREAD;
    return CE_OK;
}

static void queue_destroy(task_queue_t* q) {
    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

static void queue_push(task_queue_t* q, ce_task_t task) {
    pthread_mutex_lock(&q->lock);
    while (q->count >= TASK_QUEUE_CAP) {
        pthread_cond_wait(&q->not_full, &q->lock);
    }
    q->items[q->tail] = task;
    q->tail = (q->tail + 1) % TASK_QUEUE_CAP;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}

/* Returns 1 on success, 0 if shutdown was requested and queue is empty */
static int queue_pop(task_queue_t* q, ce_task_t* out, bool* shutdown) {
    pthread_mutex_lock(&q->lock);
    while (q->count == 0 && !(*shutdown)) {
        pthread_cond_wait(&q->not_empty, &q->lock);
    }
    if (q->count == 0 && *shutdown) {
        pthread_mutex_unlock(&q->lock);
        return 0;
    }
    *out = q->items[q->head];
    q->head = (q->head + 1) % TASK_QUEUE_CAP;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

/* ---- Thread pool ---- */

struct ce_threadpool {
    int             num_workers;
    pthread_t*      workers;
    task_queue_t    queue;
    bool            shutdown;

    /* Pending task counter for wait_all */
    _Alignas(64) _Atomic(int) pending;

    /* Barrier for wait_all */
    pthread_mutex_t wait_lock;
    pthread_cond_t  all_done;
};

static void* worker_func(void* arg) {
    ce_threadpool_t* pool = (ce_threadpool_t*)arg;
    ce_task_t task;

    while (queue_pop(&pool->queue, &task, &pool->shutdown)) {
        if (task.func) {
            task.func(task.arg);
        }
        atomic_fetch_sub(&pool->pending, 1);

        /* Wake anyone waiting in wait_all */
        pthread_mutex_lock(&pool->wait_lock);
        if (atomic_load(&pool->pending) == 0) {
            pthread_cond_signal(&pool->all_done);
        }
        pthread_mutex_unlock(&pool->wait_lock);
    }

    return NULL;
}

ce_threadpool_t* ce_threadpool_create(int num_workers) {
    if (num_workers <= 0) {
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        num_workers = (nprocs > 0) ? (int)nprocs : 4;
    }

    ce_threadpool_t* pool = calloc(1, sizeof(ce_threadpool_t));
    if (!pool) return NULL;

    pool->num_workers = num_workers;
    pool->shutdown = false;
    atomic_store(&pool->pending, 0);

    ce_status_t st = queue_init(&pool->queue);
    if (st != CE_OK) { free(pool); return NULL; }

    int rc = pthread_mutex_init(&pool->wait_lock, NULL);
    if (rc) { queue_destroy(&pool->queue); free(pool); return NULL; }
    rc = pthread_cond_init(&pool->all_done, NULL);
    if (rc) {
        pthread_mutex_destroy(&pool->wait_lock);
        queue_destroy(&pool->queue);
        free(pool);
        return NULL;
    }

    pool->workers = calloc((size_t)num_workers, sizeof(pthread_t));
    if (!pool->workers) {
        pthread_cond_destroy(&pool->all_done);
        pthread_mutex_destroy(&pool->wait_lock);
        queue_destroy(&pool->queue);
        free(pool);
        return NULL;
    }

    for (int i = 0; i < num_workers; i++) {
        rc = pthread_create(&pool->workers[i], NULL, worker_func, pool);
        if (rc) {
            /* Roll back: signal shutdown, join already-created threads */
            pool->shutdown = true;
            pthread_cond_broadcast(&pool->queue.not_empty);
            for (int j = 0; j < i; j++) {
                pthread_join(pool->workers[j], NULL);
            }
            free(pool->workers);
            pthread_cond_destroy(&pool->all_done);
            pthread_mutex_destroy(&pool->wait_lock);
            queue_destroy(&pool->queue);
            free(pool);
            return NULL;
        }

        /* Pin worker to core i (best-effort) */
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        pthread_setaffinity_np(pool->workers[i], sizeof(cpu_set_t), &cpuset);
    }

    return pool;
}

ce_status_t ce_threadpool_submit(ce_threadpool_t* pool,
                                 ce_task_func_t func, void* arg) {
    if (!pool || !func) return CE_ERR_INVALID_ARG;

    ce_task_t task = { .func = func, .arg = arg };
    atomic_fetch_add(&pool->pending, 1);
    queue_push(&pool->queue, task);
    return CE_OK;
}

ce_status_t ce_threadpool_wait_all(ce_threadpool_t* pool) {
    if (!pool) return CE_ERR_INVALID_ARG;

    pthread_mutex_lock(&pool->wait_lock);
    while (atomic_load(&pool->pending) > 0) {
        pthread_cond_wait(&pool->all_done, &pool->wait_lock);
    }
    pthread_mutex_unlock(&pool->wait_lock);

    return CE_OK;
}

void ce_threadpool_destroy(ce_threadpool_t* pool) {
    if (!pool) return;

    /* Signal shutdown */
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->queue.not_empty);

    for (int i = 0; i < pool->num_workers; i++) {
        pthread_join(pool->workers[i], NULL);
    }

    free(pool->workers);
    queue_destroy(&pool->queue);
    pthread_mutex_destroy(&pool->wait_lock);
    pthread_cond_destroy(&pool->all_done);
    free(pool);
}

int ce_threadpool_get_num_workers(ce_threadpool_t* pool) {
    if (!pool) return 0;
    return pool->num_workers;
}
