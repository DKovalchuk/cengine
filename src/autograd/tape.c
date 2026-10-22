/* tape.c — Gradient tape: dynamic array of graph nodes,
 *           records ops during forward, replay in reverse for backward */
#include "autograd.h"
#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Global tape ---- */

static ce_tape_t* g_tape = NULL;

ce_tape_t* ce_get_tape(void) {
    return g_tape;
}

void ce_set_tape(ce_tape_t* tape) {
    g_tape = tape;
}

/* ---- Lifecycle ---- */

ce_tape_t* ce_tape_create(void) {
    ce_tape_t* tape = (ce_tape_t*)calloc(1, sizeof(ce_tape_t));
    if (!tape) return NULL;

    tape->capacity = CE_TAPE_INITIAL_CAP;
    tape->entries = (tape_entry_t*)calloc(
        (size_t)tape->capacity, sizeof(tape_entry_t));
    if (!tape->entries) { free(tape); return NULL; }

    tape->length   = 0;
    tape->recording = false;
    return tape;
}

void ce_tape_destroy(ce_tape_t* tape) {
    if (!tape) return;
    for (int i = 0; i < tape->length; i++) {
        tape_entry_t* e = &tape->entries[i];
        if (e->output) ce_tensor_release(e->output);
        if (e->grad)   ce_tensor_release(e->grad);
        if (e->state)  free(e->state);
    }
    free(tape->entries);
    free(tape);
}

/* ---- Recording ---- */

ce_status_t ce_tape_start(ce_tape_t* tape) {
    if (!tape) return CE_ERR_INVALID_ARG;

    /* Clear previous entries */
    for (int i = 0; i < tape->length; i++) {
        tape_entry_t* e = &tape->entries[i];
        if (e->output) ce_tensor_release(e->output);
        if (e->grad)   ce_tensor_release(e->grad);
        if (e->state)  free(e->state);
        memset(e, 0, sizeof(tape_entry_t));
    }
    tape->length = 0;
    tape->recording = true;
    return CE_OK;
}

ce_status_t ce_tape_stop(ce_tape_t* tape) {
    if (!tape) return CE_ERR_INVALID_ARG;
    tape->recording = false;
    return CE_OK;
}

ce_status_t ce_tape_record(ce_tape_t* tape,
                            ce_tensor_t* output,
                            void (*backward_fn)(void*, ce_tensor_t*),
                            void* state, int n_inputs) {
    if (!tape || !output || !backward_fn) return CE_ERR_INVALID_ARG;
    if (!tape->recording) return CE_ERR_NOT_INITIALIZED;

    /* Grow if needed */
    if (tape->length >= tape->capacity) {
        int new_cap = tape->capacity * 2;
        tape_entry_t* new_arr = (tape_entry_t*)realloc(
            tape->entries, (size_t)new_cap * sizeof(tape_entry_t));
        if (!new_arr) return CE_ERR_OUT_OF_MEMORY;
        tape->entries = new_arr;
        memset(&tape->entries[tape->capacity], 0,
               (size_t)(new_cap - tape->capacity) * sizeof(tape_entry_t));
        tape->capacity = new_cap;
    }

    tape_entry_t* entry = &tape->entries[tape->length++];
    ce_tensor_retain(output);
    entry->output      = output;
    entry->grad        = NULL;
    entry->backward_fn = backward_fn;
    entry->state       = state;
    entry->n_inputs    = n_inputs;

    return CE_OK;
}

/* ---- Backward ---- */

ce_status_t ce_tape_backward(ce_tape_t* tape, ce_tensor_t* loss_grad) {
    if (!tape) return CE_ERR_INVALID_ARG;
    if (tape->length == 0) return CE_OK;

    /* Refuse multi-entry chains loudly.  Only the last entry ever had a
     * seeded gradient — earlier entries keep grad == NULL from record()
     * and were silently skipped — and layer tape-backwards do not
     * compute grad_input at all (linear accumulates parameter grads
     * only), so a "successful" replay returned partial gradients.
     * The production path (model->backward) is unaffected; nothing
     * in-tree called the tape backward. */
    if (tape->length > 1) {
        fprintf(stderr,
                "ce_tape_backward: %d-entry tape chains are not supported "
                "(only the last entry's op could run); use "
                "model->backward() instead\n", tape->length);
        return CE_ERR_NOT_IMPLEMENTED;
    }

    /* Seed the last entry's gradient */
    tape_entry_t* last = &tape->entries[tape->length - 1];
    if (loss_grad) {
        /* Copy loss_grad into the last entry's grad */
        if (!last->grad) {
            last->grad = ce_tensor_clone(loss_grad);
        } else {
            ce_tensor_copy(last->grad, loss_grad);
        }
    } else {
        /* Default: seed with ones (loss gradient of 1.0) */
        if (!last->grad) {
            last->grad = ce_tensor_ones(
                last->output->shape, last->output->ndim);
        }
    }

    /* Traverse in reverse order */
    for (int i = tape->length - 1; i >= 0; i--) {
        tape_entry_t* entry = &tape->entries[i];
        if (!entry->backward_fn || !entry->grad) continue;

        /* Call the backward function.
         * It receives the entry's accumulated gradient and
         * computes the gradient w.r.t. its inputs (which it may
         * propagate by modifying the gradient of earlier entries
         * via the layer's gradient tensors). */
        entry->backward_fn(entry->state, entry->grad);
    }

    return CE_OK;
}
