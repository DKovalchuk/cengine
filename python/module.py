"""
PyTorch-style module hierarchy backed by C engine layers.

Each Python layer wraps a ``ce_module_t*`` handle.  The forward / backward
passes execute entirely in C; Python owns the lifecycle and parameter bookkeeping.

Usage::

    model = Sequential(
        Linear(19, 128),
        ELU(),
        Linear(128, 128),
        ELU(),
        Linear(128, 16),
        Sigmoid(),
    )
    trainer = CEngineTrainer(model, config)
    trainer.start()
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_char_p, c_float, c_int, c_int64
from typing import Iterator

from .bindings import (
    CE_FLOAT32,
    CEngineError,
    CeModule,
    CeTensor,
    check,
    lib,
)

__all__ = [
    "Module",
    "Linear",
    "ELU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "GumbelSoftmax",
    "Sequential",
    "RecurrentExpert",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_name(name: str) -> bytes:
    """Encode a Python string to the C ``char[64]`` format."""
    return name.encode("utf-8")[:63]


def _tensor_to_numpy_view(tensor_ptr: POINTER(CeTensor)):
    """Return a zero-copy numpy view of a C tensor's data (if numpy is available)."""
    try:
        import numpy as np
    except ImportError:
        return None

    t = tensor_ptr.contents
    shape = tuple(t.shape[i] for i in range(t.ndim))
    buf = (c_float * t.numel).from_address(ctypes.addressof(t.data.contents))
    return np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()


# ---------------------------------------------------------------------------
# Module base
# ---------------------------------------------------------------------------

class Module:
    """Base class for all layers.

    Subclasses set ``self._handle`` to a ``POINTER(CeModule)`` returned by the
    C creation function.  The base class takes care of lifecycle (destroy on
    del) and parameter collection.
    """

    def __init__(self) -> None:
        self._handle: POINTER(CeModule) | None = None
        self._children: list[Module] = []
        self._name: str = ""
        self._parent: Module | None = None

    # ── Parameter access ──────────────────────────────────────────────────

    def parameters(self) -> list[POINTER(CeTensor)]:
        """Return all parameter tensors (recursively for containers)."""
        params: list[POINTER(CeTensor)] = []
        self._collect_params(params)
        return params

    def gradients(self) -> list[POINTER(CeTensor)]:
        """Return all gradient tensors (recursively for containers)."""
        grads: list[POINTER(CeTensor)] = []
        self._collect_grads(grads)
        return grads

    def _collect_params(self, out: list[POINTER(CeTensor)]) -> None:
        if self._handle is not None:
            n = lib.ce_module_param_count(self._handle)
            for i in range(n):
                out.append(lib.ce_module_get_parameter(self._handle, i))
        for child in self._children:
            child._collect_params(out)

    def _collect_grads(self, out: list[POINTER(CeTensor)]) -> None:
        if self._handle is not None:
            n = lib.ce_module_param_count(self._handle)
            for i in range(n):
                out.append(lib.ce_module_get_gradient(self._handle, i))
        for child in self._children:
            child._collect_grads(out)

    @property
    def handle(self) -> POINTER(CeModule):
        """The underlying C module handle."""
        if self._handle is None:
            raise CEngineError(f"Module {self._name!r} has not been built")
        return self._handle

    # ── Forward / backward (delegates to C) ───────────────────────────────

    def forward(self, input_tensor: POINTER(CeTensor), output_tensor: POINTER(CeTensor)) -> None:
        """Run the forward pass.  Override in leaf layers if needed."""
        raise NotImplementedError("forward() must be implemented by leaf layers")

    def backward(
        self,
        grad_output: POINTER(CeTensor),
        grad_input: POINTER(CeTensor),
    ) -> None:
        """Run the backward pass."""
        raise NotImplementedError("backward() must be implemented by leaf layers")

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the underlying C module (idempotent).

        A leaf that belongs to a container is OWNED by that container —
        closing it explicitly is a no-op, so GC of a child reference can
        never free memory the container still uses.
        """
        if self._parent is not None:
            return
        if self._handle is not None:
            lib.ce_module_destroy(self._handle)
            self._handle = None

    def _mark_closed(self) -> None:
        """Invalidate this module's and every descendant's C handle."""
        self._handle = None
        for child in self._children:
            child._mark_closed()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"


# ---------------------------------------------------------------------------
# Linear — Y = X @ W^T + b
# ---------------------------------------------------------------------------

class Linear(Module):
    """Fully-connected layer.

    Parameters are Xavier-initialised (weight) and zero-initialised (bias) by
    the C ``ce_linear_create`` function.
    """

    def __init__(self, in_features: int, out_features: int, name: str = "") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._name = name or f"linear_{in_features}x{out_features}"

        self._handle = lib.ce_linear_create(
            c_int(in_features),
            c_int(out_features),
            c_char_p(_encode_name(self._name)),
        )
        if not self._handle:
            raise CEngineError("ce_linear_create returned NULL")

    def forward(self, input_tensor: POINTER(CeTensor), output_tensor: POINTER(CeTensor)) -> None:
        check(lib.ce_linear_forward(self._handle, input_tensor, output_tensor))

    def backward(self, grad_output: POINTER(CeTensor), grad_input: POINTER(CeTensor)) -> None:
        check(lib.ce_linear_backward(self._handle, grad_output, grad_input))

    def __repr__(self) -> str:
        return f"Linear({self.in_features}, {self.out_features}, name={self._name!r})"


# ---------------------------------------------------------------------------
# Activation layers
# ---------------------------------------------------------------------------

class ELU(Module):
    """ELU activation: ``x if x > 0 else alpha * (exp(x) - 1)``."""

    def __init__(self, alpha: float = 1.0, name: str = "") -> None:
        super().__init__()
        self.alpha = alpha
        self._name = name or "elu"
        self._handle = lib.ce_elu_create(
            c_float(alpha),
            c_char_p(_encode_name(self._name)),
        )
        if not self._handle:
            raise CEngineError("ce_elu_create returned NULL")

    def forward(self, input_tensor: POINTER(CeTensor), output_tensor: POINTER(CeTensor)) -> None:
        check(lib.ce_elu_forward(self._handle, input_tensor, output_tensor))

    def backward(self, grad_output: POINTER(CeTensor), grad_input: POINTER(CeTensor)) -> None:
        check(lib.ce_elu_backward(self._handle, grad_output, grad_input))


class Sigmoid(Module):
    """Sigmoid activation: ``1 / (1 + exp(-x))``."""

    def __init__(self, name: str = "") -> None:
        super().__init__()
        self._name = name or "sigmoid"
        self._handle = lib.ce_sigmoid_create(c_char_p(_encode_name(self._name)))
        if not self._handle:
            raise CEngineError("ce_sigmoid_create returned NULL")

    def forward(self, input_tensor: POINTER(CeTensor), output_tensor: POINTER(CeTensor)) -> None:
        check(lib.ce_sigmoid_forward(self._handle, input_tensor, output_tensor))

    def backward(self, grad_output: POINTER(CeTensor), grad_input: POINTER(CeTensor)) -> None:
        check(lib.ce_sigmoid_backward(self._handle, grad_output, grad_input))


class Tanh(Module):
    """Tanh activation: ``tanh(x)``."""

    def __init__(self, name: str = "") -> None:
        super().__init__()
        self._name = name or "tanh"
        self._handle = lib.ce_tanh_create(c_char_p(_encode_name(self._name)))
        if not self._handle:
            raise CEngineError("ce_tanh_create returned NULL")

    def forward(self, input_tensor: POINTER(CeTensor), output_tensor: POINTER(CeTensor)) -> None:
        check(lib.ce_tanh_forward(self._handle, input_tensor, output_tensor))

    def backward(self, grad_output: POINTER(CeTensor), grad_input: POINTER(CeTensor)) -> None:
        check(lib.ce_tanh_backward(self._handle, grad_output, grad_input))


class Softmax(Module):
    """Softmax activation along the last dimension."""

    def __init__(self, name: str = "") -> None:
        super().__init__()
        self._name = name or "softmax"
        self._handle = lib.ce_softmax_create(c_char_p(_encode_name(self._name)))
        if not self._handle:
            raise CEngineError("ce_softmax_create returned NULL")

    def forward(self, input_tensor: POINTER(CeTensor), output_tensor: POINTER(CeTensor)) -> None:
        check(lib.ce_softmax_forward(self._handle, input_tensor, output_tensor))

    def backward(self, grad_output: POINTER(CeTensor), grad_input: POINTER(CeTensor)) -> None:
        check(lib.ce_softmax_backward(self._handle, grad_output, grad_input))


class GumbelSoftmax(Module):
    """Gumbel-Softmax with Straight-Through Estimator."""

    def __init__(self, temperature: float = 1.0, name: str = "") -> None:
        super().__init__()
        self.temperature = temperature
        self._name = name or "gumbel_softmax"
        self._handle = lib.ce_gumbel_softmax_create(
            c_float(temperature),
            c_char_p(_encode_name(self._name)),
        )
        if not self._handle:
            raise CEngineError("ce_gumbel_softmax_create returned NULL")

    def forward(self, input_tensor: POINTER(CeTensor), output_tensor: POINTER(CeTensor)) -> None:
        # Gumbel-Softmax reuses the softmax forward
        check(lib.ce_softmax_forward(self._handle, input_tensor, output_tensor))

    def backward(self, grad_output: POINTER(CeTensor), grad_input: POINTER(CeTensor)) -> None:
        check(lib.ce_softmax_backward(self._handle, grad_output, grad_input))


# ---------------------------------------------------------------------------
# Sequential container
# ---------------------------------------------------------------------------

class Sequential(Module):
    """Ordered container of layers.

    Forward passes each layer in order; backward passes in reverse.
    All children are owned by the C ``ce_sequential_t`` and destroyed when the
    container is destroyed.
    """

    def __init__(self, *layers: Module, name: str = "") -> None:
        super().__init__()
        self._name = name or "sequential"

        self._handle = lib.ce_sequential_create(c_char_p(_encode_name(self._name)))
        if not self._handle:
            raise CEngineError("ce_sequential_create returned NULL")

        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def add_module(self, name: str, module: Module) -> None:
        """Add a child module to the container."""
        check(lib.ce_sequential_add(self._handle, module.handle))
        module._parent = self
        self._children.append(module)

    def close(self) -> None:
        """Destroy the C container and its whole subtree (idempotent).

        The container owns its children (nested containers included): the
        C-side recursive destroy frees them all, then every Python handle
        pointing into the subtree is invalidated — touching them afterwards
        would be use-after-free.  Honouring _parent stops a nested
        container from tearing itself out from under its own parent.
        """
        if self._parent is not None:
            return
        if self._handle is not None:
            lib.ce_sequential_destroy(self._handle)
        self._mark_closed()

    def forward(self, input_tensor: POINTER(CeTensor), output_tensor: POINTER(CeTensor)) -> None:
        check(lib.ce_sequential_forward(self._handle, input_tensor, output_tensor))

    def backward(self, grad_output: POINTER(CeTensor), grad_input: POINTER(CeTensor)) -> None:
        check(lib.ce_sequential_backward(self._handle, grad_output, grad_input))

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, idx: int) -> Module:
        return self._children[idx]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._children)

    def __repr__(self) -> str:
        lines = [f"Sequential(name={self._name!r}:"]
        for child in self._children:
            lines.append(f"  {child!r},")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RecurrentExpert — processes sequences bit-by-bit with hidden state
# ---------------------------------------------------------------------------

class RecurrentExpert(Module):
    """Recurrent MoE ALU Expert.

    Processes binary sequences bit-by-bit (LSB to MSB) with hidden state
    and carry propagation.

    Input:  [batch, seq_len, 6] — (a_bit, b_bit, carry_slot, op_1, op_2, op_3)
    Output: [batch, seq_len, 2] — (result_logit, carry_logit)

    The C kernel indexes the input with a hardcoded stride of 6, so any
    other feature width is rejected with ``CEngineError``.
    """

    def __init__(self, hidden_size: int = 64, name: str = "recurrent_expert"):
        super().__init__()
        self.hidden_size = hidden_size
        self._name = name

        self._handle = lib.ce_recurrent_expert_create(
            c_int(hidden_size),
            c_char_p(_encode_name(name)),
        )
        if not self._handle:
            raise CEngineError("ce_recurrent_expert_create returned NULL")

    def forward(self, input_tensor: POINTER(CeTensor), output_tensor: POINTER(CeTensor)) -> None:
        # RecurrentExpert writes to output internally.
        # We need to ensure output is not freed while C references it.
        # Create a temporary output and copy to the caller's tensor.
        out_shape = (c_int64 * 3)(
            input_tensor.contents.shape[0],
            input_tensor.contents.shape[1],
            2,  # output is always [batch, T, 2]
        )
        tmp_out = lib.ce_tensor_zeros(out_shape, 3)
        try:
            check(lib.ce_recurrent_expert_forward(self._handle, input_tensor, tmp_out))
            # Copy results to caller's output
            lib.ce_tensor_copy(output_tensor, tmp_out)
        finally:
            # Released on the error path too: shape validation now raises.
            lib.ce_tensor_release(tmp_out)

    def backward(self, grad_output: POINTER(CeTensor), grad_input: POINTER(CeTensor)) -> None:
        check(lib.ce_recurrent_expert_backward(self._handle, grad_output, grad_input))

    def __repr__(self) -> str:
        return f"RecurrentExpert(hidden={self.hidden_size}, name={self._name!r})"
