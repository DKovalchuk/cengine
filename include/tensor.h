#ifndef CENGINE_TENSOR_H
#define CENGINE_TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "cengine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Max tensor dimensions */
#define CE_MAX_DIMS 8

/* Data types */
typedef enum {
    CE_FLOAT32 = 0,
    CE_FLOAT64,
    CE_INT32,
    CE_INT64,
    CE_UINT8,
} ce_dtype_t;

size_t ce_dtype_size(ce_dtype_t dtype);

/* Device */
typedef enum {
    CE_DEVICE_CPU = 0,
    /* CE_DEVICE_CUDA, */  /* future */
} ce_device_t;

/* Tensor */
typedef struct {
    float*    data;                    /* aligned pointer */
    int64_t   shape[CE_MAX_DIMS];
    int64_t   strides[CE_MAX_DIMS];
    int       ndim;
    int64_t   numel;                   /* total elements */
    size_t    nbytes;                  /* total bytes */
    ce_dtype_t dtype;
    ce_device_t device;
    bool      owns_data;               /* does this tensor own its allocation? */
    int64_t   refcount;
    char      name[64];                /* optional name for debugging */
} ce_tensor_t;

/* Creation */
ce_tensor_t* ce_tensor_create(const int64_t* shape, int ndim, ce_dtype_t dtype);
ce_tensor_t* ce_tensor_create_from_data(const float* data, const int64_t* shape, int ndim);
ce_tensor_t* ce_tensor_zeros(const int64_t* shape, int ndim);
ce_tensor_t* ce_tensor_ones(const int64_t* shape, int ndim);
ce_tensor_t* ce_tensor_from_ptr(float* ptr, const int64_t* shape, int ndim, bool owns);

/* Lifecycle */
void         ce_tensor_retain(ce_tensor_t* t);
void         ce_tensor_release(ce_tensor_t* t);

/* Shape ops */
ce_tensor_t* ce_tensor_reshape(const ce_tensor_t* t, const int64_t* new_shape, int new_ndim);
ce_tensor_t* ce_tensor_transpose(const ce_tensor_t* t, int dim0, int dim1);
ce_tensor_t* ce_tensor_slice(const ce_tensor_t* t, int dim, int64_t start, int64_t end);
ce_tensor_t* ce_tensor_clone(const ce_tensor_t* t);

/* In-place ops */
ce_status_t  ce_tensor_fill(ce_tensor_t* t, float value);
ce_status_t  ce_tensor_copy(ce_tensor_t* dst, const ce_tensor_t* src);
ce_status_t  ce_tensor_zero(ce_tensor_t* t);

/* Element access */
float*       ce_tensor_data_ptr(ce_tensor_t* t);
float        ce_tensor_get(const ce_tensor_t* t, const int64_t* indices);
void         ce_tensor_set(ce_tensor_t* t, const int64_t* indices, float value);

/* Utilities */
int64_t      ce_tensor_stride(const ce_tensor_t* t, int dim);
bool         ce_tensor_is_contiguous(const ce_tensor_t* t);
void         ce_tensor_print(const ce_tensor_t* t);

/* Multi-tensor container (for parameter lists) */
typedef struct {
    ce_tensor_t** tensors;
    int count;
    int capacity;
} ce_tensor_list_t;

ce_tensor_list_t* ce_tensor_list_create(int initial_capacity);
void              ce_tensor_list_destroy(ce_tensor_list_t* list);
ce_status_t       ce_tensor_list_append(ce_tensor_list_t* list, ce_tensor_t* t);

#ifdef __cplusplus
}
#endif

#endif /* CENGINE_TENSOR_H */
