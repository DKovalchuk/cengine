"""
Low-level ctypes bindings to libcengine.so.

Maps every public C struct to a ctypes.Structure and declares every exported
function signature with explicit argtypes / restype.  This module is the single
source of truth for the FFI boundary — all higher-level Python wrappers import
from here.

Layout rules applied:
  • _Alignas(N) fields → explicit padding bytes so the next field starts at the
    correct offset.
  • _Atomic(T) fields → same ctypes base type as T (identical ABI on Linux).
  • pthread_t → c_ulong on x86_64 Linux.
  • Function-pointer fields inside structs → c_void_p (we never call them from
    Python; the C library sets and invokes them internally).
  • Union fields → ctypes.Union subclass.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    Structure,
    Union,
    c_bool,
    c_byte,
    c_char,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int64,
    c_long,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_ulong,
    c_void_p,
)

__all__ = [
    # library handle
    "lib",
    # error handling
    "CEngineError",
    "check",
    # enums
    "CE_OK",
    "CE_FLOAT32",
    "CE_DEVICE_CPU",
    # struct types
    "CeConfig",
    "CeTensor",
    "CeTensorList",
    "CeModule",
    "CeLinear",
    "CeKanLayer",
    "CeSequential",
    "CeAdam",
    "CeSGD",
    "CeScheduler",
    "CeMemoryStats",
    "CeMetricValue",
    "CeMetric",
    "CeRingBuffer",
    "CeBarrier",
    "CeLoadBalancer",
    "CeTask",
    "CeTapeEntry",
    "CeTape",
    "CeAluSample",
    "CeBatch",
    "CeAluGen",
    "CeCurriculumStage",
    "CeCurriculum",
    "CeDataloader",
    "CeTimespec",
    "CeCollector",
    "CeTelemetrySnapshot",
    "CeReporter",
    "CePerf",
    "CeComputeBackend",
    # callback types
    "CE_TASK_FUNC_T",
]

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_LIB_SEARCH_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "lib", "libcengine.so"),
    # Fallback: LD_LIBRARY_PATH / system search
    "libcengine.so",
]

lib: ctypes.CDLL | None = None

for _p in _LIB_SEARCH_PATHS:
    _p = os.path.normpath(_p)
    try:
        lib = ctypes.CDLL(_p)
        break
    except OSError:
        continue

if lib is None:
    raise OSError(
        "Cannot load libcengine.so — build it first with:  make -C cengine shared\n"
        "Searched:\n  " + "\n  ".join(os.path.normpath(p) for p in _LIB_SEARCH_PATHS)
    )

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

CE_OK = 0

class CEngineError(RuntimeError):
    """Raised when a C engine function returns a non-zero status code."""


def check(status: int) -> None:
    """Raise CEngineError if *status* != CE_OK."""
    if status != CE_OK:
        msg_ptr = lib.ce_status_str(status)
        if msg_ptr and isinstance(msg_ptr, bytes):
            msg = msg_ptr.decode("utf-8", errors="replace")
        elif msg_ptr and isinstance(msg_ptr, int):
            import ctypes
            msg = ctypes.cast(msg_ptr, ctypes.c_char_p).value
            msg = msg.decode("utf-8", errors="replace") if msg else f"error code {status}"
        else:
            msg = f"error code {status}"
        raise CEngineError(msg)


# Convenience: expose ce_status_str
lib.ce_status_str.argtypes = [c_int]
lib.ce_status_str.restype = c_char_p

# ---------------------------------------------------------------------------
# Enum constants (subset needed by Python)
# ---------------------------------------------------------------------------

# ce_dtype_t
CE_FLOAT32 = 0
CE_FLOAT64 = 1
CE_INT32 = 2
CE_INT64 = 3
CE_UINT8 = 4

# ce_device_t
CE_DEVICE_CPU = 0

# ce_metric_type_t
CE_METRIC_LOSS_TOTAL = 0
CE_METRIC_LOSS_RESULT = 1
CE_METRIC_LOSS_CARRY = 2
CE_METRIC_LOSS_BALANCE = 3
CE_METRIC_LOSS_Z = 4
CE_METRIC_LR = 5
CE_METRIC_TEMPERATURE = 6
CE_METRIC_MEM_CURRENT = 7
CE_METRIC_MEM_PEAK = 8
CE_METRIC_MEM_GROWTH_RATE = 9
CE_METRIC_GRAD_NORM = 10
CE_METRIC_EXPERT_USAGE = 11
CE_METRIC_THROUGHPUT = 12
CE_METRIC_EPOCH_TIME_MS = 13
CE_METRIC_FORWARD_TIME_MS = 14
CE_METRIC_BACKWARD_TIME_MS = 15
CE_METRIC_OPTIMIZER_TIME_MS = 16
CE_METRIC_DATA_GEN_TIME_MS = 17
CE_METRIC_CPU_UTIL = 18
CE_METRIC_HW_IPC = 19
CE_METRIC_HW_L2_MISS_RATE = 20
CE_METRIC_HW_LLC_MISS_RATE = 21
CE_METRIC_HW_BRANCH_MISS_RATE = 22
CE_METRIC_ACTIVE_WORKERS = 23
CE_METRIC_THREADPOOL_QUEUE_DEPTH = 24
CE_METRIC_NAN_COUNT = 25
CE_METRIC_INF_COUNT = 26
CE_METRIC_CURRICULUM_STAGE = 27
CE_METRIC_COUNT = 28

CE_METRIC_FLAG_VALID = 0x01
CE_METRIC_FLAG_ALERT = 0x02

# ce_activation_t
CE_ACT_RELU = 0
CE_ACT_ELU = 1
CE_ACT_SIGMOID = 2
CE_ACT_TANH = 3
CE_ACT_SOFTMAX = 4
CE_ACT_GUMBEL_SOFTMAX = 5

# ce_transpose_t
CE_TRANSPOSE_NONE = 0
CE_TRANSPOSE_A = 1
CE_TRANSPOSE_B = 2

# ---------------------------------------------------------------------------
# Struct definitions — order matters (forward dependencies first)
# ---------------------------------------------------------------------------

# ---- ce_config_t ----
class CeConfig(Structure):
    _fields_ = [
        ("num_threads", c_int),
        ("enable_telemetry", c_bool),
        ("enable_perf_counters", c_bool),
        ("enable_debug_logging", c_bool),
        # 1 byte padding (implicit)
        ("memory_pool_size", c_size_t),
    ]


# ---- ce_tensor_t ----
class CeTensor(Structure):
    """Mirrors the C ``ce_tensor_t`` layout (248 bytes on x86_64)."""

    _fields_ = [
        ("data", POINTER(c_float)),
        ("shape", c_int64 * 8),
        ("strides", c_int64 * 8),
        ("ndim", c_int),
        # 4 bytes padding (implicit, aligns numel to 8)
        ("numel", c_int64),
        ("nbytes", c_size_t),
        ("dtype", c_int),  # ce_dtype_t enum = int
        ("device", c_int),  # ce_device_t enum = int
        ("owns_data", c_bool),
        # 7 bytes padding (implicit, aligns refcount to 8)
        ("refcount", c_int64),
        ("name", c_char * 64),
    ]


# ---- ce_tensor_list_t ----
class CeTensorList(Structure):
    _fields_ = [
        ("tensors", POINTER(POINTER(CeTensor))),
        ("count", c_int),
        ("capacity", c_int),
    ]


# ---- ce_module_t ----
class CeModule(Structure):
    """Forward-declared then completed (self-referential via fn pointers)."""

    pass


# Callback type for ce_task_func_t (threadpool)
CE_TASK_FUNC_T = ctypes.CFUNCTYPE(None, c_void_p)

CeModule._fields_ = [
    ("name", c_char * 64),
    ("parameters", POINTER(CeTensorList)),
    ("gradients", POINTER(CeTensorList)),
    ("forward", c_void_p),  # function pointer — opaque from Python
    ("backward", c_void_p),
    ("zero_grad", c_void_p),
    ("internal_state", c_void_p),
]


# ---- ce_linear_t ----
class CeLinear(Structure):
    _fields_ = [
        ("weight", POINTER(CeTensor)),
        ("bias", POINTER(CeTensor)),
        ("grad_w", POINTER(CeTensor)),
        ("grad_b", POINTER(CeTensor)),
        ("input_cache", POINTER(CeTensor)),
        ("in_features", c_int),
        ("out_features", c_int),
        ("name", c_char * 64),
    ]


# ---- ce_kan_layer_t ----
class CeKanLayer(Structure):
    _fields_ = [
        ("w_b", POINTER(CeTensor)),
        ("w_s", POINTER(CeTensor)),
        ("c", POINTER(CeTensor)),
        ("bias", POINTER(CeTensor)),
        ("g_w_b", POINTER(CeTensor)),
        ("g_w_s", POINTER(CeTensor)),
        ("g_c", POINTER(CeTensor)),
        ("g_bias", POINTER(CeTensor)),
        ("saved_basis", POINTER(CeTensor)),
        ("saved_silu", POINTER(CeTensor)),
        ("saved_x_norm", POINTER(CeTensor)),
        ("saved_c64", POINTER(CeTensor)),
        ("in_features", c_int),
        ("out_features", c_int),
        ("name", c_char * 64),
    ]


# ---- ce_recurrent_expert_t ----
class CeRecurrentExpert(Structure):
    _fields_ = [
        ("w1", POINTER(CeTensor)), ("b1", POINTER(CeTensor)),
        ("w2", POINTER(CeTensor)), ("b2", POINTER(CeTensor)),
        ("w_out", POINTER(CeTensor)), ("b_out", POINTER(CeTensor)),
        ("w_h", POINTER(CeTensor)), ("b_h", POINTER(CeTensor)),
        ("gw1", POINTER(CeTensor)), ("gb1", POINTER(CeTensor)),
        ("gw2", POINTER(CeTensor)), ("gb2", POINTER(CeTensor)),
        ("gw_out", POINTER(CeTensor)), ("gb_out", POINTER(CeTensor)),
        ("gw_h", POINTER(CeTensor)), ("gb_h", POINTER(CeTensor)),
        ("saved_inputs", POINTER(CeTensor)),
        ("saved_hiddens", POINTER(CeTensor)),
        ("saved_carry", POINTER(CeTensor)),
        ("saved_logits", POINTER(CeTensor)),
        ("saved_tanh_pre", POINTER(CeTensor)),
        ("saved_elu1", POINTER(CeTensor)),
        ("saved_elu2", POINTER(CeTensor)),
        ("hidden_size", c_int),
        ("input_dim", c_int),
        ("name", c_char * 64),
    ]


# ---- ce_sequential_t ----
class CeSequential(Structure):
    _fields_ = [
        ("children", POINTER(POINTER(CeModule))),
        ("count", c_int),
        ("capacity", c_int),
    ]


# ---- ce_adam_t ----
class CeAdam(Structure):
    _fields_ = [
        ("lr", c_float),
        ("beta1", c_float),
        ("beta2", c_float),
        ("eps", c_float),
        ("weight_decay", c_float),
        # 4 bytes padding (implicit, aligns pointer to 8)
        ("params", POINTER(POINTER(CeTensor))),
        ("grads", POINTER(POINTER(CeTensor))),
        ("m", POINTER(POINTER(CeTensor))),
        ("v", POINTER(POINTER(CeTensor))),
        ("n_params", c_int),
        ("step", c_int),
    ]


# ---- ce_sgd_t ----
class CeSGD(Structure):
    _fields_ = [
        ("lr", c_float),
        ("momentum", c_float),
        ("weight_decay", c_float),
        # 4 bytes padding (implicit)
        ("params", POINTER(POINTER(CeTensor))),
        ("grads", POINTER(POINTER(CeTensor))),
        ("velocity", POINTER(POINTER(CeTensor))),
        ("n_params", c_int),
    ]


# ---- ce_scheduler_t ----
class CeScheduler(Structure):
    _fields_ = [
        ("max_lr", c_float),
        ("min_lr", c_float),
        ("pct_start", c_float),
        ("div_factor", c_float),
        ("final_div_factor", c_float),
        ("total_steps", c_int),
        ("current_step", c_int),
        ("current_lr", c_float),
    ]


# ---- ce_memory_stats_t ----
class CeMemoryStats(Structure):
    """Returned by value from ``ce_tracker_get_stats``."""

    _fields_ = [
        ("current_bytes", c_int64),
        ("peak_bytes", c_int64),
        ("total_allocs", c_int64),
        ("total_frees", c_int64),
        ("active_allocs", c_int64),
        ("alloc_bytes_rate", c_int64),
        ("free_bytes_rate", c_int64),
        ("growth_rate", c_double),
        ("fragmentation", c_double),
        ("tensor_count", c_int64),
        ("tensor_total_bytes", c_int64),
    ]


# ---- ce_metric_t (with union) ----
class CeMetricValue(Union):
    _fields_ = [
        ("f32", c_float),
        ("f32_arr", c_float * 8),
        ("i64", c_int64),
    ]


class CeMetric(Structure):
    """48-byte metric record pushed into the ring buffer."""

    _fields_ = [
        ("metric_type", c_uint8),
        ("flags", c_uint8),
        ("_pad", c_uint16),
        ("epoch", c_uint32),
        ("timestamp_ns", c_uint64),
        ("value", CeMetricValue),
    ]


# ---- ce_ring_buffer_t (196 736 bytes) ----
# head/tail are _Alignas(64) _Atomic(uint64_t).
# buffer is _Alignas(64) ce_metric_t[4096].
class CeRingBuffer(Structure):
    _fields_ = [
        ("head", c_uint64),
        ("_pad1", c_byte * 56),  # pad to offset 64
        ("tail", c_uint64),
        ("_pad2", c_byte * 56),  # pad to offset 128
        ("buffer", CeMetric * 4096),
    ]


# ---- ce_barrier_t ----
class CeBarrier(Structure):
    _fields_ = [
        ("count", c_int),
        ("_pad1", c_byte * 60),  # _Alignas(64)
        ("generation", c_int),
        ("_pad2", c_byte * 60),  # _Alignas(64)
        ("num_threads", c_int),
    ]


# ---- ce_load_balancer_t ----
class CeLoadBalancer(Structure):
    _fields_ = [
        ("min_workers", c_int),
        ("max_workers", c_int),
        ("target_cpu_util", c_float),
        ("current_workers", c_int),
        ("prev_idle", c_uint64),
        ("prev_total", c_uint64),
    ]


# ---- ce_task_t ----
class CeTask(Structure):
    _fields_ = [
        ("func", c_void_p),  # ce_task_func_t — opaque from Python
        ("arg", c_void_p),
    ]


# ---- autograd structs ----
class CeTapeEntry(Structure):
    _fields_ = [
        ("output", POINTER(CeTensor)),
        ("grad", POINTER(CeTensor)),
        ("backward_fn", c_void_p),
        ("state", c_void_p),
        ("n_inputs", c_int),
    ]


class CeTape(Structure):
    _fields_ = [
        ("entries", POINTER(CeTapeEntry)),
        ("length", c_int),
        ("capacity", c_int),
        ("recording", c_bool),
    ]


# ---- data structs ----
class CeAluSample(Structure):
    _fields_ = [
        ("a_bits", POINTER(CeTensor)),
        ("b_bits", POINTER(CeTensor)),
        ("op_code", POINTER(CeTensor)),
        ("result_bits", POINTER(CeTensor)),
        ("carry_bits", POINTER(CeTensor)),
        ("n_bits", c_int),
        ("op", c_int),
    ]


class CeBatch(Structure):
    _fields_ = [
        ("samples", POINTER(POINTER(CeAluSample))),
        ("count", c_int),
        ("capacity", c_int),
    ]


class CeAluGen(Structure):
    _fields_ = [
        ("min_bits", c_int),
        ("max_bits", c_int),
        ("ops", c_int * 5),  # CE_NUM_OPS = 5
        ("n_ops", c_int),
        # 4 bytes padding (implicit, aligns seed to 8)
        ("seed", c_int64),
        ("rng_state", c_uint64),
    ]


class CeCurriculumStage(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("min_bits", c_int),
        ("max_bits", c_int),
        ("ops", c_int * 5),  # CE_NUM_OPS = 5
        ("n_ops", c_int),
    ]


class CeCurriculum(Structure):
    _fields_ = [
        ("stages", POINTER(CeCurriculumStage)),
        ("n_stages", c_int),
        ("current_stage", c_int),
        ("epochs_in_stage", c_int),
        ("consecutive_good", c_int),
        ("loss_threshold", c_float),
        ("patience", c_int),
        ("max_epochs_per_stage", c_int),
    ]


class CeDataloader(Structure):
    _fields_ = [
        ("generator", POINTER(CeAluGen)),
        ("batch_size", c_int),
        ("samples_per_epoch", c_int),
        ("base_seed", c_int64),
        ("epoch", c_int64),
        ("prefetch_buffer", POINTER(CeBatch) * 2),
        ("ready_buffer", c_int),  # _Atomic(int) — same ABI
        # 4 bytes padding (implicit, aligns pthread_t)
        ("prefetch_thread", c_ulong),  # pthread_t on x86_64
        ("stop_flag", c_bool),  # _Atomic(bool)
        ("thread_active", c_bool),
    ]


# ---- telemetry structs ----
class CeTimespec(Structure):
    _fields_ = [
        ("tv_sec", c_long),
        ("tv_nsec", c_long),
    ]


class CeCollector(Structure):
    _fields_ = [
        ("ring", POINTER(CeRingBuffer)),
        ("epoch_start", CeTimespec),
        ("forward_start", CeTimespec),
        ("backward_start", CeTimespec),
        ("optimizer_start", CeTimespec),
        ("data_gen_start", CeTimespec),
        ("nan_count", c_int64),
        ("inf_count", c_int64),
        ("current_epoch", c_int),
        # 4 bytes padding (implicit)
        ("samples_processed", c_int64),
    ]


class CeTelemetrySnapshot(Structure):
    """Flat, pointer-free snapshot — ideal for by-value copy into Python."""

    _fields_ = [
        ("loss_total", c_float),
        ("loss_result", c_float),
        ("loss_carry", c_float),
        ("loss_balance", c_float),
        ("loss_z", c_float),
        ("lr", c_float),
        ("temperature", c_float),
        ("mem_current_mb", c_float),
        ("mem_peak_mb", c_float),
        ("mem_growth_rate", c_float),
        ("grad_norm", c_float),
        ("expert_usage", c_float * 4),
        ("throughput", c_float),
        ("epoch_time_ms", c_float),
        ("forward_time_ms", c_float),
        ("backward_time_ms", c_float),
        ("optimizer_time_ms", c_float),
        ("data_gen_time_ms", c_float),
        ("cpu_util", c_float),
        ("hw_ipc", c_float),
        ("hw_l2_miss_rate", c_float),
        ("hw_llc_miss_rate", c_float),
        ("hw_branch_miss_rate", c_float),
        ("active_workers", c_int),
        ("threadpool_queue_depth", c_int),
        ("nan_count", c_int64),
        ("inf_count", c_int64),
        ("curriculum_stage", c_int),
        ("current_epoch", c_int),
        ("last_update_ns", c_uint64),
    ]


class CeReporter(Structure):
    _fields_ = [
        ("ring", POINTER(CeRingBuffer)),
        ("snapshot", CeTelemetrySnapshot),
    ]


class CePerf(Structure):
    _fields_ = [
        ("fd_instructions", c_int),
        ("fd_cycles", c_int),
        ("fd_cache_references", c_int),
        ("fd_cache_misses", c_int),
        ("fd_branch_instructions", c_int),
        ("fd_branch_misses", c_int),
        ("available", c_bool),
    ]


# ---- compute backend vtable ----
class CeComputeBackend(Structure):
    """Opaque vtable — we only need the pointer, not the fields."""

    pass


CeComputeBackend._fields_ = [
    ("name", c_char_p),
    ("device", c_int),
    ("matmul", c_void_p),
    ("add", c_void_p),
    ("mul", c_void_p),
    ("axpy", c_void_p),
    ("activation", c_void_p),
    ("activation_backward", c_void_p),
    ("sum", c_void_p),
    ("mean", c_void_p),
    ("max", c_void_p),
    ("norm", c_void_p),
    ("dot", c_void_p),
    ("get_max_threads", c_void_p),
]


# ---------------------------------------------------------------------------
# Function declarations — grouped by header
# ---------------------------------------------------------------------------

# ---- cengine.h ----
lib.ce_init.argtypes = []
lib.ce_init.restype = c_int

lib.ce_shutdown.argtypes = []
lib.ce_shutdown.restype = None

lib.ce_configure.argtypes = [POINTER(CeConfig)]
lib.ce_configure.restype = c_int

lib.ce_get_config.argtypes = []
lib.ce_get_config.restype = POINTER(CeConfig)

# ---- tensor.h ----
lib.ce_dtype_size.argtypes = [c_int]
lib.ce_dtype_size.restype = c_size_t

lib.ce_tensor_create.argtypes = [POINTER(c_int64), c_int, c_int]
lib.ce_tensor_create.restype = POINTER(CeTensor)

lib.ce_tensor_create_from_data.argtypes = [POINTER(c_float), POINTER(c_int64), c_int]
lib.ce_tensor_create_from_data.restype = POINTER(CeTensor)

lib.ce_tensor_zeros.argtypes = [POINTER(c_int64), c_int]
lib.ce_tensor_zeros.restype = POINTER(CeTensor)

lib.ce_tensor_ones.argtypes = [POINTER(c_int64), c_int]
lib.ce_tensor_ones.restype = POINTER(CeTensor)

lib.ce_tensor_from_ptr.argtypes = [POINTER(c_float), POINTER(c_int64), c_int, c_bool]
lib.ce_tensor_from_ptr.restype = POINTER(CeTensor)

lib.ce_tensor_retain.argtypes = [POINTER(CeTensor)]
lib.ce_tensor_retain.restype = None

lib.ce_tensor_release.argtypes = [POINTER(CeTensor)]
lib.ce_tensor_release.restype = None

lib.ce_tensor_reshape.argtypes = [POINTER(CeTensor), POINTER(c_int64), c_int]
lib.ce_tensor_reshape.restype = POINTER(CeTensor)

lib.ce_tensor_transpose.argtypes = [POINTER(CeTensor), c_int, c_int]
lib.ce_tensor_transpose.restype = POINTER(CeTensor)

lib.ce_tensor_slice.argtypes = [POINTER(CeTensor), c_int, c_int64, c_int64]
lib.ce_tensor_slice.restype = POINTER(CeTensor)

lib.ce_tensor_clone.argtypes = [POINTER(CeTensor)]
lib.ce_tensor_clone.restype = POINTER(CeTensor)

lib.ce_tensor_fill.argtypes = [POINTER(CeTensor), c_float]
lib.ce_tensor_fill.restype = c_int

lib.ce_tensor_copy.argtypes = [POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_tensor_copy.restype = c_int

lib.ce_tensor_zero.argtypes = [POINTER(CeTensor)]
lib.ce_tensor_zero.restype = c_int

lib.ce_tensor_data_ptr.argtypes = [POINTER(CeTensor)]
lib.ce_tensor_data_ptr.restype = POINTER(c_float)

lib.ce_tensor_get.argtypes = [POINTER(CeTensor), POINTER(c_int64)]
lib.ce_tensor_get.restype = c_float

lib.ce_tensor_set.argtypes = [POINTER(CeTensor), POINTER(c_int64), c_float]
lib.ce_tensor_set.restype = None

lib.ce_tensor_stride.argtypes = [POINTER(CeTensor), c_int]
lib.ce_tensor_stride.restype = c_int64

lib.ce_tensor_is_contiguous.argtypes = [POINTER(CeTensor)]
lib.ce_tensor_is_contiguous.restype = c_bool

lib.ce_tensor_print.argtypes = [POINTER(CeTensor)]
lib.ce_tensor_print.restype = None

lib.ce_tensor_list_create.argtypes = [c_int]
lib.ce_tensor_list_create.restype = POINTER(CeTensorList)

lib.ce_tensor_list_destroy.argtypes = [POINTER(CeTensorList)]
lib.ce_tensor_list_destroy.restype = None

lib.ce_tensor_list_append.argtypes = [POINTER(CeTensorList), POINTER(CeTensor)]
lib.ce_tensor_list_append.restype = c_int

# ---- layers.h ----
lib.ce_module_create.argtypes = [c_char_p]
lib.ce_module_create.restype = POINTER(CeModule)

lib.ce_module_destroy.argtypes = [POINTER(CeModule)]
lib.ce_module_destroy.restype = None

lib.ce_module_add_parameter.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_module_add_parameter.restype = c_int

lib.ce_module_get_parameter.argtypes = [POINTER(CeModule), c_int]
lib.ce_module_get_parameter.restype = POINTER(CeTensor)

lib.ce_module_get_gradient.argtypes = [POINTER(CeModule), c_int]
lib.ce_module_get_gradient.restype = POINTER(CeTensor)

lib.ce_module_param_count.argtypes = [POINTER(CeModule)]
lib.ce_module_param_count.restype = c_int

lib.ce_module_default_zero_grad.argtypes = [POINTER(CeModule)]
lib.ce_module_default_zero_grad.restype = None

lib.ce_clip_module_grads.argtypes = [POINTER(CeModule), c_float]
lib.ce_clip_module_grads.restype = c_int

# Linear
lib.ce_linear_create.argtypes = [c_int, c_int, c_char_p]
lib.ce_linear_create.restype = POINTER(CeModule)

lib.ce_linear_forward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_linear_forward.restype = c_int

lib.ce_linear_backward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_linear_backward.restype = c_int

# KAN Layer (Polynomial Basis + SiLU)
lib.ce_kan_layer_create.argtypes = [c_int, c_int, c_char_p]
lib.ce_kan_layer_create.restype = POINTER(CeModule)

lib.ce_kan_layer_forward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_kan_layer_forward.restype = c_int

lib.ce_kan_layer_backward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_kan_layer_backward.restype = c_int

lib.ce_kan_layer_forward_batch.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_kan_layer_forward_batch.restype = c_int

lib.ce_kan_layer_backward_batch.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_kan_layer_backward_batch.restype = c_int

# Recurrent KAN
lib.ce_kan_recurrent_create.argtypes = [c_int, c_char_p]
lib.ce_kan_recurrent_create.restype = POINTER(CeModule)

lib.ce_kan_recurrent_forward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_kan_recurrent_forward.restype = c_int

# ELU
lib.ce_elu_create.argtypes = [c_float, c_char_p]
lib.ce_elu_create.restype = POINTER(CeModule)

lib.ce_elu_forward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_elu_forward.restype = c_int

lib.ce_elu_backward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_elu_backward.restype = c_int

# Sigmoid
lib.ce_sigmoid_create.argtypes = [c_char_p]
lib.ce_sigmoid_create.restype = POINTER(CeModule)

lib.ce_sigmoid_forward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_sigmoid_forward.restype = c_int

lib.ce_sigmoid_backward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_sigmoid_backward.restype = c_int

# Tanh
lib.ce_tanh_create.argtypes = [c_char_p]
lib.ce_tanh_create.restype = POINTER(CeModule)

lib.ce_tanh_forward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_tanh_forward.restype = c_int

lib.ce_tanh_backward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_tanh_backward.restype = c_int

# Softmax / Gumbel-Softmax
lib.ce_softmax_create.argtypes = [c_char_p]
lib.ce_softmax_create.restype = POINTER(CeModule)

lib.ce_gumbel_softmax_create.argtypes = [c_float, c_char_p]
lib.ce_gumbel_softmax_create.restype = POINTER(CeModule)

lib.ce_softmax_forward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_softmax_forward.restype = c_int

lib.ce_softmax_backward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_softmax_backward.restype = c_int

# Sequential
lib.ce_sequential_create.argtypes = [c_char_p]
lib.ce_sequential_create.restype = POINTER(CeModule)

lib.ce_sequential_add.argtypes = [POINTER(CeModule), POINTER(CeModule)]
lib.ce_sequential_add.restype = c_int

lib.ce_sequential_forward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_sequential_forward.restype = c_int

lib.ce_sequential_backward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_sequential_backward.restype = c_int

# ---- recurrent.h ----
lib.ce_recurrent_expert_create.argtypes = [c_int, c_char_p]
lib.ce_recurrent_expert_create.restype = POINTER(CeModule)

lib.ce_recurrent_expert_forward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_recurrent_expert_forward.restype = c_int

lib.ce_recurrent_expert_forward_infer.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_recurrent_expert_forward_infer.restype = c_int

# Pre-alloc variant: hidden_buf, carry_buf, x5h_buf, h_new_buf, elu1_buf, elu2_buf
lib.ce_recurrent_expert_forward_infer_prealloc.argtypes = [
    POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
lib.ce_recurrent_expert_forward_infer_prealloc.restype = c_int

lib.ce_recurrent_expert_backward.argtypes = [POINTER(CeModule), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_recurrent_expert_backward.restype = c_int

lib.ce_recurrent_expert_forward_routed.argtypes = [
    POINTER(POINTER(CeModule)), c_int,
    POINTER(CeTensor), POINTER(CeTensor),
    ctypes.POINTER(c_int),
]
lib.ce_recurrent_expert_forward_routed.restype = c_int

# ---- optimizer.h ----
lib.ce_adam_create.argtypes = [
    c_float, c_float, c_float, c_float, c_float,
    POINTER(POINTER(CeTensor)),
    POINTER(POINTER(CeTensor)),
    c_int,
]
lib.ce_adam_create.restype = POINTER(CeAdam)

lib.ce_adam_step.argtypes = [POINTER(CeAdam)]
lib.ce_adam_step.restype = c_int

lib.ce_adam_destroy.argtypes = [POINTER(CeAdam)]
lib.ce_adam_destroy.restype = None

lib.ce_sgd_create.argtypes = [
    c_float, c_float, c_float,
    POINTER(POINTER(CeTensor)),
    POINTER(POINTER(CeTensor)),
    c_int,
]
lib.ce_sgd_create.restype = POINTER(CeSGD)

lib.ce_sgd_step.argtypes = [POINTER(CeSGD)]
lib.ce_sgd_step.restype = c_int

lib.ce_sgd_destroy.argtypes = [POINTER(CeSGD)]
lib.ce_sgd_destroy.restype = None

lib.ce_scheduler_create.argtypes = [c_float, c_float, c_float, c_float, c_int]
lib.ce_scheduler_create.restype = POINTER(CeScheduler)

lib.ce_scheduler_step.argtypes = [POINTER(CeScheduler)]
lib.ce_scheduler_step.restype = c_float

lib.ce_scheduler_reset.argtypes = [POINTER(CeScheduler)]
lib.ce_scheduler_reset.restype = None

lib.ce_scheduler_destroy.argtypes = [POINTER(CeScheduler)]
lib.ce_scheduler_destroy.restype = None

lib.ce_compute_grad_norm.argtypes = [POINTER(POINTER(CeTensor)), c_int]
lib.ce_compute_grad_norm.restype = c_float

lib.ce_clip_grad_norm.argtypes = [POINTER(POINTER(CeTensor)), c_int, c_float]
lib.ce_clip_grad_norm.restype = c_int

# ---- loss.h ----
lib.ce_bce_with_logits_loss.argtypes = [
    POINTER(CeTensor), POINTER(CeTensor), POINTER(CeTensor),
]
lib.ce_bce_with_logits_loss.restype = c_float

lib.ce_bce_with_logits_backward.argtypes = [
    POINTER(CeTensor), POINTER(CeTensor), POINTER(CeTensor), POINTER(CeTensor),
]
lib.ce_bce_with_logits_backward.restype = c_int

lib.ce_mse_loss.argtypes = [POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_mse_loss.restype = c_float

lib.ce_mse_backward.argtypes = [POINTER(CeTensor), POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_mse_backward.restype = c_int

# ---- memory.h ----
lib.ce_aligned_alloc.argtypes = [c_size_t, c_size_t]
lib.ce_aligned_alloc.restype = c_void_p

lib.ce_aligned_free.argtypes = [c_void_p]
lib.ce_aligned_free.restype = None

# Pool (opaque)
lib.ce_mem_pool_create.argtypes = [c_size_t]
lib.ce_mem_pool_create.restype = c_void_p

lib.ce_mem_pool_destroy.argtypes = [c_void_p]
lib.ce_mem_pool_destroy.restype = None

lib.ce_mem_pool_alloc.argtypes = [c_void_p, c_size_t]
lib.ce_mem_pool_alloc.restype = c_void_p

lib.ce_mem_pool_free.argtypes = [c_void_p, c_void_p]
lib.ce_mem_pool_free.restype = None

lib.ce_mem_pool_reset.argtypes = [c_void_p]
lib.ce_mem_pool_reset.restype = None

lib.ce_tracker_init.argtypes = []
lib.ce_tracker_init.restype = c_int

lib.ce_tracker_shutdown.argtypes = []
lib.ce_tracker_shutdown.restype = None

lib.ce_tracker_record_alloc.argtypes = [c_void_p, c_size_t, c_char_p]
lib.ce_tracker_record_alloc.restype = None

lib.ce_tracker_record_free.argtypes = [c_void_p]
lib.ce_tracker_record_free.restype = None

lib.ce_tracker_update_tensor.argtypes = [c_int64, c_int64]
lib.ce_tracker_update_tensor.restype = None

lib.ce_tracker_get_stats.argtypes = []
lib.ce_tracker_get_stats.restype = CeMemoryStats  # returned by value

lib.ce_tracker_reset_rates.argtypes = []
lib.ce_tracker_reset_rates.restype = None

lib.ce_tracker_check_leaks.argtypes = []
lib.ce_tracker_check_leaks.restype = c_int

# ---- autograd.h ----
lib.ce_tape_create.argtypes = []
lib.ce_tape_create.restype = POINTER(CeTape)

lib.ce_tape_destroy.argtypes = [POINTER(CeTape)]
lib.ce_tape_destroy.restype = None

lib.ce_tape_start.argtypes = [POINTER(CeTape)]
lib.ce_tape_start.restype = c_int

lib.ce_tape_stop.argtypes = [POINTER(CeTape)]
lib.ce_tape_stop.restype = c_int

lib.ce_tape_backward.argtypes = [POINTER(CeTape), POINTER(CeTensor)]
lib.ce_tape_backward.restype = c_int

lib.ce_get_tape.argtypes = []
lib.ce_get_tape.restype = POINTER(CeTape)

lib.ce_set_tape.argtypes = [POINTER(CeTape)]
lib.ce_set_tape.restype = None

lib.ce_backward.argtypes = [POINTER(CeTensor)]
lib.ce_backward.restype = c_int

lib.ce_compute_grad.argtypes = [POINTER(CeTensor), POINTER(CeTensor)]
lib.ce_compute_grad.restype = c_int

# ---- data.h ----
lib.ce_alu_gen_create.argtypes = [c_int, c_int, POINTER(c_int), c_int, c_int64]
lib.ce_alu_gen_create.restype = POINTER(CeAluGen)

lib.ce_alu_gen_sample.argtypes = [POINTER(CeAluGen)]
lib.ce_alu_gen_sample.restype = POINTER(CeAluSample)

lib.ce_alu_gen_batch.argtypes = [POINTER(CeAluGen), c_int]
lib.ce_alu_gen_batch.restype = POINTER(CeBatch)

lib.ce_alu_sample_destroy.argtypes = [POINTER(CeAluSample)]
lib.ce_alu_sample_destroy.restype = None

lib.ce_batch_destroy.argtypes = [POINTER(CeBatch)]
lib.ce_batch_destroy.restype = None

lib.ce_curriculum_create.argtypes = [c_float, c_int, c_int, c_int]
lib.ce_curriculum_create.restype = POINTER(CeCurriculum)

lib.ce_curriculum_update.argtypes = [POINTER(CeCurriculum), c_float]
lib.ce_curriculum_update.restype = c_int

lib.ce_curriculum_make_generator.argtypes = [POINTER(CeCurriculum), c_int64]
lib.ce_curriculum_make_generator.restype = POINTER(CeAluGen)

lib.ce_curriculum_stage_name.argtypes = [POINTER(CeCurriculum)]
lib.ce_curriculum_stage_name.restype = c_char_p

lib.ce_curriculum_is_final.argtypes = [POINTER(CeCurriculum)]
lib.ce_curriculum_is_final.restype = c_bool

lib.ce_curriculum_destroy.argtypes = [POINTER(CeCurriculum)]
lib.ce_curriculum_destroy.restype = None

lib.ce_dataloader_create.argtypes = [POINTER(CeAluGen), c_int, c_int, c_int64]
lib.ce_dataloader_create.restype = POINTER(CeDataloader)

lib.ce_dataloader_next.argtypes = [POINTER(CeDataloader)]
lib.ce_dataloader_next.restype = POINTER(CeBatch)

lib.ce_dataloader_reset.argtypes = [POINTER(CeDataloader)]
lib.ce_dataloader_reset.restype = None

lib.ce_dataloader_destroy.argtypes = [POINTER(CeDataloader)]
lib.ce_dataloader_destroy.restype = None

# ---- Checkpoint ----
lib.ce_checkpoint_save.argtypes = [c_char_p, c_int, c_float, POINTER(CeModule), c_void_p]
lib.ce_checkpoint_save.restype = c_int

lib.ce_checkpoint_load.argtypes = [c_char_p, POINTER(CeModule)]
lib.ce_checkpoint_load.restype = c_int

lib.ce_bits_le.argtypes = [c_uint64, c_int, POINTER(c_float)]
lib.ce_bits_le.restype = None

# ---- threadpool.h ----
lib.ce_threadpool_create.argtypes = [c_int]
lib.ce_threadpool_create.restype = c_void_p  # opaque

lib.ce_threadpool_submit.argtypes = [c_void_p, c_void_p, c_void_p]
lib.ce_threadpool_submit.restype = c_int

lib.ce_threadpool_wait_all.argtypes = [c_void_p]
lib.ce_threadpool_wait_all.restype = c_int

lib.ce_threadpool_destroy.argtypes = [c_void_p]
lib.ce_threadpool_destroy.restype = None

lib.ce_threadpool_get_num_workers.argtypes = [c_void_p]
lib.ce_threadpool_get_num_workers.restype = c_int

lib.ce_barrier_init.argtypes = [POINTER(CeBarrier), c_int]
lib.ce_barrier_init.restype = c_int

lib.ce_barrier_wait.argtypes = [POINTER(CeBarrier)]
lib.ce_barrier_wait.restype = None

lib.ce_barrier_destroy.argtypes = [POINTER(CeBarrier)]
lib.ce_barrier_destroy.restype = None

lib.ce_ring_push.argtypes = [POINTER(CeRingBuffer), POINTER(CeMetric)]
lib.ce_ring_push.restype = c_int

lib.ce_ring_pop.argtypes = [POINTER(CeRingBuffer), POINTER(CeMetric)]
lib.ce_ring_pop.restype = c_int

lib.ce_ring_batch_pop.argtypes = [POINTER(CeRingBuffer), POINTER(CeMetric), c_int]
lib.ce_ring_batch_pop.restype = c_int

lib.ce_load_balancer_create.argtypes = [c_int, c_int, c_float]
lib.ce_load_balancer_create.restype = POINTER(CeLoadBalancer)

lib.ce_load_balancer_adjust.argtypes = [POINTER(CeLoadBalancer)]
lib.ce_load_balancer_adjust.restype = c_int

lib.ce_load_balancer_destroy.argtypes = [POINTER(CeLoadBalancer)]
lib.ce_load_balancer_destroy.restype = None

# ---- telemetry.h ----
lib.ce_collector_create.argtypes = [POINTER(CeRingBuffer)]
lib.ce_collector_create.restype = POINTER(CeCollector)

lib.ce_collector_destroy.argtypes = [POINTER(CeCollector)]
lib.ce_collector_destroy.restype = None

lib.ce_collector_start_epoch.argtypes = [POINTER(CeCollector), c_int]
lib.ce_collector_start_epoch.restype = None

lib.ce_collector_end_epoch.argtypes = [POINTER(CeCollector)]
lib.ce_collector_end_epoch.restype = None

lib.ce_collector_start_forward.argtypes = [POINTER(CeCollector)]
lib.ce_collector_start_forward.restype = None

lib.ce_collector_end_forward.argtypes = [POINTER(CeCollector)]
lib.ce_collector_end_forward.restype = None

lib.ce_collector_start_backward.argtypes = [POINTER(CeCollector)]
lib.ce_collector_start_backward.restype = None

lib.ce_collector_end_backward.argtypes = [POINTER(CeCollector)]
lib.ce_collector_end_backward.restype = None

lib.ce_collector_start_optimizer.argtypes = [POINTER(CeCollector)]
lib.ce_collector_start_optimizer.restype = None

lib.ce_collector_end_optimizer.argtypes = [POINTER(CeCollector)]
lib.ce_collector_end_optimizer.restype = None

lib.ce_collector_start_data_gen.argtypes = [POINTER(CeCollector)]
lib.ce_collector_start_data_gen.restype = None

lib.ce_collector_end_data_gen.argtypes = [POINTER(CeCollector)]
lib.ce_collector_end_data_gen.restype = None

lib.ce_collector_record_loss.argtypes = [
    POINTER(CeCollector), c_float, c_float, c_float, c_float, c_float,
]
lib.ce_collector_record_loss.restype = None

lib.ce_collector_record_lr.argtypes = [POINTER(CeCollector), c_float, c_float]
lib.ce_collector_record_lr.restype = None

lib.ce_collector_record_memory.argtypes = [POINTER(CeCollector), POINTER(CeMemoryStats)]
lib.ce_collector_record_memory.restype = None

lib.ce_collector_record_grad_norm.argtypes = [POINTER(CeCollector), c_float]
lib.ce_collector_record_grad_norm.restype = None

lib.ce_collector_record_expert_usage.argtypes = [POINTER(CeCollector), POINTER(c_float)]
lib.ce_collector_record_expert_usage.restype = None

lib.ce_collector_record_throughput.argtypes = [POINTER(CeCollector), c_float]
lib.ce_collector_record_throughput.restype = None

lib.ce_collector_record_anomalies.argtypes = [POINTER(CeCollector), c_int64, c_int64]
lib.ce_collector_record_anomalies.restype = None

lib.ce_collector_record_curriculum.argtypes = [POINTER(CeCollector), c_int]
lib.ce_collector_record_curriculum.restype = None

lib.ce_reporter_create.argtypes = [POINTER(CeRingBuffer)]
lib.ce_reporter_create.restype = POINTER(CeReporter)

lib.ce_reporter_destroy.argtypes = [POINTER(CeReporter)]
lib.ce_reporter_destroy.restype = None

lib.ce_reporter_poll.argtypes = [POINTER(CeReporter)]
lib.ce_reporter_poll.restype = c_int

lib.ce_reporter_get_snapshot.argtypes = [POINTER(CeReporter)]
lib.ce_reporter_get_snapshot.restype = CeTelemetrySnapshot  # by value

lib.ce_perf_create.argtypes = []
lib.ce_perf_create.restype = POINTER(CePerf)

lib.ce_perf_destroy.argtypes = [POINTER(CePerf)]
lib.ce_perf_destroy.restype = None

lib.ce_perf_read.argtypes = [
    POINTER(CePerf), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
]
lib.ce_perf_read.restype = None

# ---- compute.h ----
lib.ce_register_backend.argtypes = [POINTER(CeComputeBackend)]
lib.ce_register_backend.restype = c_int

lib.ce_get_backend.argtypes = [c_int]
lib.ce_get_backend.restype = POINTER(CeComputeBackend)

lib.ce_get_default_backend.argtypes = []
lib.ce_get_default_backend.restype = POINTER(CeComputeBackend)
