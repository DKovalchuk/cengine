/*
 * trainer_internal.h — internal trainer helpers, exposed for unit tests.
 *
 * Not part of the public C API; not installed with include/.
 */
#ifndef CENGINE_TRAINER_INTERNAL_H
#define CENGINE_TRAINER_INTERNAL_H

#include "trainer.h"

void ce_trainer_prepare_batch(const ce_batch_t* batch, int max_bits,
                              ce_tensor_t* input, ce_tensor_t* target,
                              ce_tensor_t* mask, int input_dim);

#endif /* CENGINE_TRAINER_INTERNAL_H */
