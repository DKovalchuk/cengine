/*
 * curriculum.c — Curriculum learning manager
 *
 * Stages mirror data.py's DEFAULT_CURRICULUM:
 *   0: "Warm-up: short ADD+XOR"        4-8  bits, {ADD, XOR}
 *   1: "Stage 1: ADD+SUB+XOR"          4-16 bits, {ADD, SUB, XOR}
 *   2: "Stage 2: all ops, medium"       4-24 bits, {ADD, SUB, AND, XOR}
 *   3: "Stage 3: all ops, long"         8-32 bits, {ADD, SUB, AND, XOR}
 */
#include "data.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* Default curriculum table */
static ce_curriculum_stage_t g_default_stages[] = {
    {
        .name     = "Warm-up: short ADD+XOR",
        .min_bits = 4,  .max_bits = 8,
        .ops      = { CE_OP_ADD, CE_OP_XOR },
        .n_ops    = 2,
    },
    {
        .name     = "Stage 1: ADD+SUB+XOR",
        .min_bits = 4,  .max_bits = 16,
        .ops      = { CE_OP_ADD, CE_OP_SUB, CE_OP_XOR },
        .n_ops    = 3,
    },
    {
        .name     = "Stage 2: all ops, medium",
        .min_bits = 4,  .max_bits = 24,
        .ops      = { CE_OP_ADD, CE_OP_SUB, CE_OP_AND, CE_OP_XOR },
        .n_ops    = 4,
    },
    {
        .name     = "Stage 3: all ops, long",
        .min_bits = 8,  .max_bits = 32,
        .ops      = { CE_OP_ADD, CE_OP_SUB, CE_OP_AND, CE_OP_XOR },
        .n_ops    = 4,
    },
};

#define CE_DEFAULT_STAGE_COUNT 4

/* ------------------------------------------------------------------ */

ce_curriculum_t* ce_curriculum_create(float loss_threshold, int patience,
                                      int max_epochs, int max_bits) {
    ce_curriculum_t* c = calloc(1, sizeof(*c));
    if (!c) return NULL;

    /* Allocate and copy default stages */
    c->n_stages = CE_DEFAULT_STAGE_COUNT;
    c->stages = calloc((size_t)c->n_stages, sizeof(ce_curriculum_stage_t));
    if (!c->stages) { free(c); return NULL; }
    memcpy(c->stages, g_default_stages,
           sizeof(ce_curriculum_stage_t) * (size_t)c->n_stages);

    /* Cap stages to max_bits: remove stages that exceed the limit,
       and trim the last included stage to max_bits. */
    if (max_bits > 0) {
        int keep = 0;
        for (int i = 0; i < c->n_stages; i++) {
            if (c->stages[i].min_bits > max_bits) {
                break;  /* stage starts beyond limit — stop here */
            }
            keep = i + 1;
            /* Cap this stage's max_bits */
            if (c->stages[i].max_bits > max_bits) {
                c->stages[i].max_bits = max_bits;
            }
            /* Also cap min_bits so it doesn't exceed max_bits */
            if (c->stages[i].min_bits > max_bits) {
                c->stages[i].min_bits = max_bits;
            }
        }
        if (keep > 0 && keep < c->n_stages) {
            c->n_stages = keep;
        }
    }

    c->current_stage       = 0;
    c->epochs_in_stage     = 0;
    c->consecutive_good    = 0;
    c->loss_threshold      = loss_threshold;
    c->patience            = patience;
    c->max_epochs_per_stage = max_epochs;

    return c;
}

int ce_curriculum_update(ce_curriculum_t* c, float epoch_loss) {
    if (!c) return 0;

    c->epochs_in_stage++;

    if (epoch_loss < c->loss_threshold) {
        c->consecutive_good++;
    } else {
        c->consecutive_good = 0;
    }

    int advanced = 0;
    if (!ce_curriculum_is_final(c)) {
        if (c->consecutive_good >= c->patience ||
            c->epochs_in_stage >= c->max_epochs_per_stage) {
            c->current_stage++;
            c->epochs_in_stage  = 0;
            c->consecutive_good = 0;
            advanced = 1;
        }
    }

    return advanced;
}

ce_alu_gen_t* ce_curriculum_make_generator(ce_curriculum_t* c, int64_t seed) {
    if (!c || c->current_stage < 0 || c->current_stage >= c->n_stages) {
        return NULL;
    }
    ce_curriculum_stage_t* s = &c->stages[c->current_stage];
    return ce_alu_gen_create(s->min_bits, s->max_bits,
                             s->ops, s->n_ops, seed);
}

const char* ce_curriculum_stage_name(const ce_curriculum_t* c) {
    if (!c || c->current_stage < 0 || c->current_stage >= c->n_stages) {
        return "invalid";
    }
    return c->stages[c->current_stage].name;
}

bool ce_curriculum_is_final(const ce_curriculum_t* c) {
    if (!c) return true;
    return c->current_stage >= c->n_stages - 1;
}

void ce_curriculum_destroy(ce_curriculum_t* c) {
    if (!c) return;
    free(c->stages);
    free(c);
}
