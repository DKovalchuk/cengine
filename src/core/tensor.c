#include "tensor.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

size_t ce_dtype_size(ce_dtype_t dtype) {
    switch (dtype) {
        case CE_FLOAT32: return 4;
        case CE_FLOAT64: return 8;
        case CE_INT32:   return 4;
        case CE_INT64:   return 8;
        case CE_UINT8:   return 1;
        default:         return 0;
    }
}

/* Compute strides from shape (row-major) */
static void compute_strides(int64_t* strides, const int64_t* shape, int ndim) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

ce_tensor_t* ce_tensor_create(const int64_t* shape, int ndim, ce_dtype_t dtype) {
    if (!shape || ndim <= 0 || ndim > CE_MAX_DIMS) return NULL;

    ce_tensor_t* t = (ce_tensor_t*)calloc(1, sizeof(ce_tensor_t));
    if (!t) return NULL;

    t->ndim = ndim;
    t->dtype = dtype;
    t->device = CE_DEVICE_CPU;
    t->refcount = 1;
    t->owns_data = true;

    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    compute_strides(t->strides, shape, ndim);

    t->nbytes = (size_t)t->numel * ce_dtype_size(dtype);
    t->data = (float*)ce_aligned_alloc(CE_SIMD_ALIGNMENT, t->nbytes);
    if (!t->data) {
        free(t);
        return NULL;
    }
    memset(t->data, 0, t->nbytes);

    ce_tracker_update_tensor(1, (int64_t)t->nbytes);
    return t;
}

ce_tensor_t* ce_tensor_create_from_data(const float* data, const int64_t* shape, int ndim) {
    ce_tensor_t* t = ce_tensor_create(shape, ndim, CE_FLOAT32);
    if (!t) return NULL;
    if (data) {
        memcpy(t->data, data, t->nbytes);
    }
    return t;
}

ce_tensor_t* ce_tensor_zeros(const int64_t* shape, int ndim) {
    /* ce_tensor_create already zeros memory */
    return ce_tensor_create(shape, ndim, CE_FLOAT32);
}

ce_tensor_t* ce_tensor_ones(const int64_t* shape, int ndim) {
    ce_tensor_t* t = ce_tensor_create(shape, ndim, CE_FLOAT32);
    if (!t) return NULL;
    for (int64_t i = 0; i < t->numel; i++) {
        t->data[i] = 1.0f;
    }
    return t;
}

ce_tensor_t* ce_tensor_from_ptr(float* ptr, const int64_t* shape, int ndim, bool owns) {
    if (!ptr || !shape || ndim <= 0) return NULL;

    ce_tensor_t* t = (ce_tensor_t*)calloc(1, sizeof(ce_tensor_t));
    if (!t) return NULL;

    t->ndim = ndim;
    t->dtype = CE_FLOAT32;
    t->device = CE_DEVICE_CPU;
    t->refcount = 1;
    t->owns_data = owns;
    t->data = ptr;

    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    compute_strides(t->strides, shape, ndim);
    t->nbytes = (size_t)t->numel * sizeof(float);

    ce_tracker_update_tensor(1, (int64_t)t->nbytes);
    return t;
}

void ce_tensor_retain(ce_tensor_t* t) {
    if (t) t->refcount++;
}

void ce_tensor_release(ce_tensor_t* t) {
    if (!t) return;
    t->refcount--;
    if (t->refcount <= 0) {
        ce_tracker_update_tensor(-1, -(int64_t)t->nbytes);
        if (t->owns_data && t->data) {
            ce_aligned_free(t->data);
        }
        free(t);
    }
}

ce_tensor_t* ce_tensor_reshape(const ce_tensor_t* t, const int64_t* new_shape, int new_ndim) {
    if (!t || !new_shape || new_ndim <= 0) return NULL;

    int64_t new_numel = 1;
    for (int i = 0; i < new_ndim; i++) new_numel *= new_shape[i];
    if (new_numel != t->numel) return NULL;

    /* Create a view (doesn't own data) */
    ce_tensor_t* view = (ce_tensor_t*)calloc(1, sizeof(ce_tensor_t));
    if (!view) return NULL;

    view->data = t->data;
    view->ndim = new_ndim;
    view->dtype = t->dtype;
    view->device = t->device;
    view->numel = t->numel;
    view->nbytes = t->nbytes;
    view->owns_data = false;
    view->refcount = 1;

    for (int i = 0; i < new_ndim; i++) view->shape[i] = new_shape[i];
    compute_strides(view->strides, new_shape, new_ndim);

    return view;
}

ce_tensor_t* ce_tensor_transpose(const ce_tensor_t* t, int dim0, int dim1) {
    if (!t || dim0 < 0 || dim1 < 0 || dim0 >= t->ndim || dim1 >= t->ndim) return NULL;

    ce_tensor_t* view = (ce_tensor_t*)calloc(1, sizeof(ce_tensor_t));
    if (!view) return NULL;

    view->data = t->data;
    view->ndim = t->ndim;
    view->dtype = t->dtype;
    view->device = t->device;
    view->numel = t->numel;
    view->nbytes = t->nbytes;
    view->owns_data = false;
    view->refcount = 1;

    memcpy(view->shape, t->shape, sizeof(int64_t) * t->ndim);
    memcpy(view->strides, t->strides, sizeof(int64_t) * t->ndim);

    /* Swap shape and stride for the two dimensions */
    int64_t tmp;
    tmp = view->shape[dim0]; view->shape[dim0] = view->shape[dim1]; view->shape[dim1] = tmp;
    tmp = view->strides[dim0]; view->strides[dim0] = view->strides[dim1]; view->strides[dim1] = tmp;

    return view;
}

ce_tensor_t* ce_tensor_slice(const ce_tensor_t* t, int dim, int64_t start, int64_t end) {
    if (!t || dim < 0 || dim >= t->ndim) return NULL;
    if (start < 0 || end > t->shape[dim] || start >= end) return NULL;

    ce_tensor_t* view = (ce_tensor_t*)calloc(1, sizeof(ce_tensor_t));
    if (!view) return NULL;

    view->data = t->data + start * t->strides[dim];
    view->ndim = t->ndim;
    view->dtype = t->dtype;
    view->device = t->device;
    view->owns_data = false;
    view->refcount = 1;

    memcpy(view->shape, t->shape, sizeof(int64_t) * t->ndim);
    memcpy(view->strides, t->strides, sizeof(int64_t) * t->ndim);
    view->shape[dim] = end - start;

    view->numel = 1;
    for (int i = 0; i < view->ndim; i++) view->numel *= view->shape[i];
    view->nbytes = (size_t)view->numel * ce_dtype_size(view->dtype);

    return view;
}

ce_tensor_t* ce_tensor_clone(const ce_tensor_t* t) {
    if (!t) return NULL;
    ce_tensor_t* c = ce_tensor_create(t->shape, t->ndim, t->dtype);
    if (!c) return NULL;
    memcpy(c->data, t->data, t->nbytes);
    return c;
}

ce_status_t ce_tensor_fill(ce_tensor_t* t, float value) {
    if (!t || !t->data) return CE_ERR_INVALID_ARG;
    for (int64_t i = 0; i < t->numel; i++) t->data[i] = value;
    return CE_OK;
}

ce_status_t ce_tensor_copy(ce_tensor_t* dst, const ce_tensor_t* src) {
    if (!dst || !src) return CE_ERR_INVALID_ARG;
    if (dst->numel != src->numel) return CE_ERR_INVALID_SHAPE;
    memcpy(dst->data, src->data, src->nbytes);
    return CE_OK;
}

ce_status_t ce_tensor_zero(ce_tensor_t* t) {
    if (!t || !t->data) return CE_ERR_INVALID_ARG;
    memset(t->data, 0, t->nbytes);
    return CE_OK;
}

float* ce_tensor_data_ptr(ce_tensor_t* t) {
    return t ? t->data : NULL;
}

float ce_tensor_get(const ce_tensor_t* t, const int64_t* indices) {
    if (!t || !indices) return 0.0f;
    int64_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    return t->data[offset];
}

void ce_tensor_set(ce_tensor_t* t, const int64_t* indices, float value) {
    if (!t || !indices) return;
    int64_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    t->data[offset] = value;
}

int64_t ce_tensor_stride(const ce_tensor_t* t, int dim) {
    if (!t || dim < 0 || dim >= t->ndim) return 0;
    return t->strides[dim];
}

bool ce_tensor_is_contiguous(const ce_tensor_t* t) {
    if (!t) return false;
    int64_t expected_stride = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected_stride) return false;
        expected_stride *= t->shape[i];
    }
    return true;
}

void ce_tensor_print(const ce_tensor_t* t) {
    if (!t) { printf("Tensor(NULL)\n"); return; }
    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%ld%s", (long)t->shape[i], (i < t->ndim - 1) ? "," : "");
    }
    printf("], numel=%ld, dtype=%d, ptr=%p, refs=%ld)\n",
           (long)t->numel, t->dtype, (void*)t->data, (long)t->refcount);
}

/* ---- Tensor list ---- */

ce_tensor_list_t* ce_tensor_list_create(int initial_capacity) {
    if (initial_capacity <= 0) initial_capacity = 8;
    ce_tensor_list_t* list = (ce_tensor_list_t*)calloc(1, sizeof(ce_tensor_list_t));
    if (!list) return NULL;
    list->tensors = (ce_tensor_t**)calloc((size_t)initial_capacity, sizeof(ce_tensor_t*));
    if (!list->tensors) { free(list); return NULL; }
    list->count = 0;
    list->capacity = initial_capacity;
    return list;
}

void ce_tensor_list_destroy(ce_tensor_list_t* list) {
    if (!list) return;
    /* Release all tensors */
    for (int i = 0; i < list->count; i++) {
        ce_tensor_release(list->tensors[i]);
    }
    free(list->tensors);
    free(list);
}

ce_status_t ce_tensor_list_append(ce_tensor_list_t* list, ce_tensor_t* t) {
    if (!list || !t) return CE_ERR_INVALID_ARG;
    if (list->count >= list->capacity) {
        int new_cap = list->capacity * 2;
        ce_tensor_t** new_arr = (ce_tensor_t**)realloc(
            list->tensors, (size_t)new_cap * sizeof(ce_tensor_t*));
        if (!new_arr) return CE_ERR_OUT_OF_MEMORY;
        list->tensors = new_arr;
        list->capacity = new_cap;
    }
    ce_tensor_retain(t);
    list->tensors[list->count++] = t;
    return CE_OK;
}
