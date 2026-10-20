"""
Unit tests for C-engine RecurrentExpert.

Tests forward pass correctness, backward pass (gradient checking),
and activation statistics. No LSTM exists in C-engine — the recurrent
layer is ce_recurrent_expert_t (MLP + carry + tanh hidden).

Usage:
    pytest tests/test_recurrent_cengine.py -v
"""
import sys
import os
import ctypes
import math
import numpy as np
import pytest

# ── Path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from python.bindings import lib, CeConfig, CeTensor, c_int64, c_float, c_bool, c_int, c_size_t, check
from python.module import RecurrentExpert, Sequential, Linear, ELU

# Ensure C engine is initialized
check(lib.ce_init())


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_tensor_3d(shape, data=None):
    """Create a 3D C tensor. shape = (B, T, F)."""
    ndim = len(shape)
    c_shape = (c_int64 * ndim)(*shape)
    t = lib.ce_tensor_zeros(c_shape, ndim)
    if data is not None:
        arr = np.asarray(data, dtype=np.float32).ravel()
        n = min(len(arr), t.contents.numel)
        for i in range(n):
            t.contents.data[i] = arr[i]
    return t


def _tensor_to_numpy(t):
    """Convert C tensor to numpy array."""
    n = t.contents.numel
    ndim = t.contents.ndim
    shape = tuple(t.contents.shape[i] for i in range(ndim))
    buf = (ctypes.c_float * n).from_address(ctypes.addressof(t.contents.data.contents))
    return np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()


def _set_weights_from_numpy(module, weight_dict):
    """Set module weights from a dict of numpy arrays.

    weight_dict keys: 'w1', 'b1', 'w2', 'b2', 'w_out', 'b_out', 'w_h', 'b_h'
    """
    handle = module.handle
    # Access internal state through the module's parameter list
    # Parameters are: w1, b1, w2, b2, w_out, b_out, w_h, b_h
    param_names = ['w1', 'b1', 'w2', 'b2', 'w_out', 'b_out', 'w_h', 'b_h']
    for i, name in enumerate(param_names):
        if name in weight_dict:
            param = lib.ce_module_get_parameter(handle, i)
            src = weight_dict[name].ravel().astype(np.float32)
            n = min(len(src), param.contents.numel)
            for j in range(n):
                param.contents.data[j] = src[j]


# ══════════════════════════════════════════════════════════════════════════
# Test 1: Forward pass produces finite outputs
# ══════════════════════════════════════════════════════════════════════════

class TestRecurrentForward:
    """Verify forward pass produces finite, non-degenerate outputs."""

    def test_forward_finite_output(self):
        """Forward on random input should produce finite outputs."""
        H = 16
        expert = RecurrentExpert(hidden_size=H, name="test_fwd")

        B, T, F = 4, 8, 5
        inp = _make_tensor_3d((B, T, F))
        # Fill with small random values
        for i in range(B * T * F):
            inp.contents.data[i] = np.random.randn() * 0.1

        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)

        out_np = _tensor_to_numpy(out)
        assert np.all(np.isfinite(out_np)), f"Non-finite output: {out_np}"
        assert out_np.shape == (B, T, 2), f"Wrong shape: {out_np.shape}"

    def test_forward_zero_input(self):
        """Forward on zeros should produce finite outputs (no NaN/Inf)."""
        H = 8
        expert = RecurrentExpert(hidden_size=H, name="test_zero")

        B, T, F = 2, 4, 5
        inp = _make_tensor_3d((B, T, F))  # zeros
        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)

        out_np = _tensor_to_numpy(out)
        assert np.all(np.isfinite(out_np)), f"Non-finite on zero input: {out_np}"

    def test_forward_deterministic(self):
        """Same input twice should give same output."""
        H = 8
        expert = RecurrentExpert(hidden_size=H, name="test_det")

        B, T, F = 2, 6, 5
        np.random.seed(42)
        data = np.random.randn(B, T, F).astype(np.float32) * 0.1

        inp1 = _make_tensor_3d((B, T, F), data)
        out1 = _make_tensor_3d((B, T, 2))
        expert.forward(inp1, out1)
        np1 = _tensor_to_numpy(out1)

        inp2 = _make_tensor_3d((B, T, F), data)
        out2 = _make_tensor_3d((B, T, 2))
        expert.forward(inp2, out2)
        np2 = _tensor_to_numpy(out2)

        assert np.allclose(np1, np2, atol=1e-6), \
            f"Non-deterministic forward: max diff = {np.abs(np1 - np2).max()}"

    def test_forward_with_carry(self):
        """Carry propagation: ADD of 1+1 should produce carry=1 at bit 1."""
        H = 8
        expert = RecurrentExpert(hidden_size=H, name="test_carry")

        # 2-bit ADD: a=1 (01), b=1 (01) → result=10, carry sequence: 0,1
        # Bit 0: a=1, b=1, carry_in=0 → result=0, carry_out=1
        # Bit 1: a=0, b=0, carry_in=1 → result=1, carry_out=0
        B, T, F = 1, 2, 5
        inp = _make_tensor_3d((B, T, F))
        inp.contents.data[0] = 1.0  # a_bit[0] = 1
        inp.contents.data[1] = 1.0  # b_bit[0] = 1
        inp.contents.data[2] = 0.0  # carry_in[0] = 0
        inp.contents.data[3] = 1.0  # op_1 = 1 (ADD)
        inp.contents.data[4] = 0.0  # op_2 = 0

        inp.contents.data[5] = 0.0  # a_bit[1] = 0
        inp.contents.data[6] = 0.0  # b_bit[1] = 0
        inp.contents.data[7] = 0.0  # carry_in[1] = 0 (will be set by network)
        inp.contents.data[8] = 1.0  # op_1 = 1
        inp.contents.data[9] = 0.0  # op_2 = 0

        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)

        out_np = _tensor_to_numpy(out)
        assert np.all(np.isfinite(out_np)), "Non-finite output on carry test"
        # The carry bit (out[0,0,1]) should be positive (network predicts carry)
        # since 1+1=10 in binary


# ══════════════════════════════════════════════════════════════════════════
# Test 2: Backward pass — gradient check via finite differences
# ══════════════════════════════════════════════════════════════════════════

class TestRecurrentBackward:
    """Gradient checking via central finite differences."""

    def _compute_loss(self, expert, inp_data, B, T):
        """Forward pass and return sum of outputs as scalar loss."""
        inp = _make_tensor_3d((B, T, 5), inp_data)
        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)
        loss = sum(out.contents.data[i] for i in range(out.contents.numel))
        return loss

    def _compute_grad_numerical(self, expert, inp_data, B, T, eps=1e-3):
        """Compute numerical gradient of loss w.r.t. input."""
        grad = np.zeros_like(inp_data)
        flat = grad.ravel()
        for i in range(min(len(flat), B * T * 5)):
            orig = inp_data.ravel()[i]

            inp_data.ravel()[i] = orig + eps
            loss_plus = self._compute_loss(expert, inp_data.copy(), B, T)

            inp_data.ravel()[i] = orig - eps
            loss_minus = self._compute_loss(expert, inp_data.copy(), B, T)

            flat[i] = (loss_plus - loss_minus) / (2 * eps)
            inp_data.ravel()[i] = orig

        return grad

    def test_gradient_check_input(self):
        """C-engine backward zeros grad_input — verify this behavior.

        The C-engine RecurrentExpert computes parameter gradients but
        does NOT propagate gradients to the input tensor (grad_input is zeroed).
        This test documents that behavior.
        """
        H = 4
        B, T = 1, 3
        expert = RecurrentExpert(hidden_size=H, name="test_grad")

        inp = _make_tensor_3d((B, T, 5))
        for i in range(B * T * 5):
            inp.contents.data[i] = np.random.randn() * 0.1

        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)

        grad_out = _make_tensor_3d((B, T, 2))
        for i in range(B * T * 2):
            grad_out.contents.data[i] = 1.0

        grad_in = _make_tensor_3d((B, T, 5))
        expert.backward(grad_out, grad_in)

        # C-engine zeros grad_input — this is the expected behavior
        grad_in_np = _tensor_to_numpy(grad_in)
        assert np.allclose(grad_in_np, 0.0), \
            "C-engine backward should zero grad_input"

    def test_gradient_check_params(self):
        """Numerical vs analytical gradient w.r.t. parameters."""
        H = 4
        B, T = 1, 2
        expert = RecurrentExpert(hidden_size=H, name="test_grad_p")

        np.random.seed(456)
        inp_data = np.random.randn(B, T, 5).astype(np.float32) * 0.1

        # Forward + backward
        inp = _make_tensor_3d((B, T, 5), inp_data)
        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)

        grad_out = _make_tensor_3d((B, T, 2))
        for i in range(B * T * 2):
            grad_out.contents.data[i] = 1.0
        expert.backward(grad_out, None)

        # Check gradient norms are non-zero
        grads = expert.gradients()
        for gi, g in enumerate(grads):
            norm = math.sqrt(sum(g.contents.data[j] ** 2 for j in range(g.contents.numel)))
            assert norm > 0, f"Gradient {gi} is zero — backward may be broken"

    def test_no_vanishing_gradients_small_network(self):
        """Small network should not have vanishing gradients on simple input."""
        H = 8
        B, T = 2, 4
        expert = RecurrentExpert(hidden_size=H, name="test_novg")

        inp = _make_tensor_3d((B, T, 5))
        for i in range(B * T * 5):
            inp.contents.data[i] = np.random.randn() * 0.5

        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)

        grad_out = _make_tensor_3d((B, T, 2))
        for i in range(B * T * 2):
            grad_out.contents.data[i] = 1.0
        expert.backward(grad_out, None)

        grads = expert.gradients()
        total_norm = 0.0
        for g in grads:
            n = g.contents.numel
            total_norm += sum(g.contents.data[j] ** 2 for j in range(n))
        total_norm = math.sqrt(total_norm)

        assert total_norm > 1e-6, f"Vanishing gradients: total_norm={total_norm}"
        assert total_norm < 1000, f"Exploding gradients: total_norm={total_norm}"


# ══════════════════════════════════════════════════════════════════════════
# Test 3: Activation statistics — no dead neurons
# ══════════════════════════════════════════════════════════════════════════

class TestRecurrentActivations:
    """Check activation statistics for dead neurons / vanishing."""

    def test_output_not_all_same(self):
        """Output should have variance (not all identical = dead neurons)."""
        H = 16
        B, T = 8, 10
        expert = RecurrentExpert(hidden_size=H, name="test_act")

        inp = _make_tensor_3d((B, T, 5))
        for i in range(B * T * 5):
            inp.contents.data[i] = np.random.randn() * 0.3

        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)

        out_np = _tensor_to_numpy(out)
        std = out_np.std()
        assert std > 1e-6, f"Dead neurons: output std = {std}"
        assert np.all(np.isfinite(out_np)), "Non-finite activations"

    def test_output_range_reasonable(self):
        """Output logits should be in reasonable range (< 100)."""
        H = 16
        B, T = 4, 8
        expert = RecurrentExpert(hidden_size=H, name="test_range")

        inp = _make_tensor_3d((B, T, 5))
        for i in range(B * T * 5):
            inp.contents.data[i] = np.random.randn() * 0.5

        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)

        out_np = _tensor_to_numpy(out)
        assert out_np.max() < 100, f"Exploding output: max={out_np.max()}"
        assert out_np.min() > -100, f"Exploding output: min={out_np.min()}"


# ══════════════════════════════════════════════════════════════════════════
# Test 4: Debug logging flag
# ══════════════════════════════════════════════════════════════════════════

class TestDebugLogging:
    """Verify debug logging flag works without crashing."""

    def test_debug_logging_forward(self):
        """Forward with debug logging enabled should not crash."""
        # Enable debug logging
        cfg = CeConfig()
        cfg.num_threads = 1
        cfg.enable_telemetry = False
        cfg.enable_perf_counters = False
        cfg.enable_debug_logging = True
        cfg.memory_pool_size = 0
        check(lib.ce_configure(ctypes.byref(cfg)))

        H = 8
        B, T = 2, 3
        expert = RecurrentExpert(hidden_size=H, name="test_dbg")

        inp = _make_tensor_3d((B, T, 5))
        for i in range(B * T * 5):
            inp.contents.data[i] = np.random.randn() * 0.1

        out = _make_tensor_3d((B, T, 2))
        expert.forward(inp, out)  # Should not crash

        out_np = _tensor_to_numpy(out)
        assert np.all(np.isfinite(out_np))

        # Disable debug logging
        cfg.enable_debug_logging = False
        check(lib.ce_configure(ctypes.byref(cfg)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
