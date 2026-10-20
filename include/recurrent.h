#ifndef CENGINE_RECURRENT_H
#define CENGINE_RECURRENT_H

#include "layers.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Recurrent MoE ALU Expert
 *
 * Processes binary sequences bit-by-bit (LSB to MSB) with hidden state
 * and carry propagation, exactly like the PyTorch version.
 *
 * Per timestep t:
 *   input:  [a_bit, b_bit, carry_in, op_1, op_2] + hidden_state → [5+H]
 *   MLP:    Linear(5+H, H) -> ELU -> Linear(H, H) -> ELU
 *   output: head_out(H→2) -> [result_logit, carry_logit]
 *   hidden: head_h(H→H) -> tanh -> new_hidden
 *   carry:  sigmoid(carry_logit) -> carry_in for next step
 *
 * Input tensor:  [batch, seq_len, 5] — (a_bit, b_bit, carry_in, op_1, op_2)
 * Output tensor: [batch, seq_len, 2] — (result_logit, carry_logit)
 * ================================================================ */

typedef struct {
    /* MLP weights */
    ce_tensor_t* w1;        /* [H, 5+H] — first linear */
    ce_tensor_t* b1;        /* [H] */
    ce_tensor_t* w2;        /* [H, H] — second linear */
    ce_tensor_t* b2;        /* [H] */
    ce_tensor_t* w_out;     /* [2, H] — output head */
    ce_tensor_t* b_out;     /* [2] */
    ce_tensor_t* w_h;       /* [H, H] — hidden head */
    ce_tensor_t* b_h;       /* [H] */

    /* Gradient accumulators */
    ce_tensor_t* gw1, *gb1, *gw2, *gb2;
    ce_tensor_t* gw_out, *gb_out, *gw_h, *gb_h;

    /* Saved state for backward (BPTT) */
    ce_tensor_t* saved_inputs;    /* [batch, seq_len, 5+H] */
    ce_tensor_t* saved_hiddens;   /* [batch, seq_len+1, H] — post-tanh hidden */
    ce_tensor_t* saved_carry;     /* [batch, seq_len] */
    ce_tensor_t* saved_logits;    /* [batch, seq_len, 2] */

    int hidden_size;
    int input_dim;         /* always 6 */
    char name[64];
} ce_recurrent_expert_t;

ce_module_t* ce_recurrent_expert_create(int hidden_size, const char* name);
ce_status_t  ce_recurrent_expert_forward(ce_module_t* mod,
                                          ce_tensor_t* input,
                                          ce_tensor_t* output);
ce_status_t  ce_recurrent_expert_forward_infer(ce_module_t* mod,
                                                ce_tensor_t* input,
                                                ce_tensor_t* output);

/* Forward (inference with pre-allocated buffers: zero malloc in hot loop) */
ce_status_t  ce_recurrent_expert_forward_infer_prealloc(
    ce_module_t* mod, ce_tensor_t* input, ce_tensor_t* output,
    float* hidden_buf, float* carry_buf,
    float* x5h_buf, float* h_new_buf,
    float* elu1_buf, float* elu2_buf);

ce_status_t  ce_recurrent_expert_backward(ce_module_t* mod,
                                           ce_tensor_t* grad_output,
                                           ce_tensor_t* grad_input);

/* MoE routing: process batch split by expert */
ce_status_t  ce_recurrent_expert_forward_routed(ce_module_t** experts,
                                                  int num_experts,
                                                  ce_tensor_t* input,
                                                  ce_tensor_t* output,
                                                  int* assignments);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_RECURRENT_H */
