/* graph.c — Computation graph node: output tensor, grad tensor,
 *            backward function pointer, input references */
#include "autograd.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>

/* Create a graph node.  Graph nodes are the elements that the tape
 * stores; the tape itself manages the array.  This function is
 * exposed for advanced use cases (manual graph construction). */
tape_entry_t* ce_graph_node_create(ce_tensor_t* output,
                                    void (*backward_fn)(void*, ce_tensor_t*),
                                    void* state, int n_inputs) {
    if (!output || !backward_fn) return NULL;

    tape_entry_t* node = (tape_entry_t*)calloc(1, sizeof(tape_entry_t));
    if (!node) return NULL;

    node->output      = output;
    node->grad        = NULL;
    node->backward_fn = backward_fn;
    node->state       = state;
    node->n_inputs    = n_inputs;

    return node;
}

void ce_graph_node_destroy(tape_entry_t* node) {
    if (!node) return;
    if (node->output) ce_tensor_release(node->output);
    if (node->grad)   ce_tensor_release(node->grad);
    if (node->state)  free(node->state);
    free(node);
}

/* Allocate a gradient tensor for the given output (same shape). */
ce_status_t ce_graph_node_ensure_grad(tape_entry_t* node) {
    if (!node || !node->output) return CE_ERR_INVALID_ARG;
    if (node->grad) return CE_OK;  /* already allocated */

    node->grad = ce_tensor_zeros(node->output->shape, node->output->ndim);
    return node->grad ? CE_OK : CE_ERR_OUT_OF_MEMORY;
}

ce_tensor_t* ce_graph_node_get_grad(const tape_entry_t* node) {
    return node ? node->grad : NULL;
}

ce_tensor_t* ce_graph_node_get_output(const tape_entry_t* node) {
    return node ? node->output : NULL;
}

/* Build adjacency: for a given node, return which prior nodes
 * produced its inputs.  This is a helper for topological sort.
 * NOTE: requires that the state object's first pointer field
 * points to the relevant input tape entry (convention). */
int ce_graph_count_dependencies(const tape_entry_t* node) {
    return node ? node->n_inputs : 0;
}
