/*
 * trainer_internal.h — internal trainer helpers, exposed for unit tests.
 *
 * Not part of the public C API; not installed with include/.
 */
#ifndef CENGINE_TRAINER_INTERNAL_H
#define CENGINE_TRAINER_INTERNAL_H

#include "trainer.h"

void ce_trainer_prepare_batch(const ce_batch_t* batch,
                              ce_tensor_t* input, ce_tensor_t* target,
                              ce_tensor_t* mask, int input_dim);

/* Test-only accessors into trainer run state. */
int   ce_trainer_curriculum_stage(const ce_trainer_t* t);   /* -1 if not initialised */
float ce_trainer_current_loss(const ce_trainer_t* t);

#endif /* CENGINE_TRAINER_INTERNAL_H */
