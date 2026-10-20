#ifndef CENGINE_KAN_RECURRENT_H
#define CENGINE_KAN_RECURRENT_H

#include "layers.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Recurrent KAN ALU: carry loop + 3 KAN layers */
ce_module_t* ce_kan_recurrent_create(int hidden_size, const char* name);
ce_status_t  ce_kan_recurrent_forward(ce_module_t* mod,
                                       ce_tensor_t* input,
                                       ce_tensor_t* output);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_KAN_RECURRENT_H */
