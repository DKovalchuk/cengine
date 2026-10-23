"""
C-checkpoint must roundtrip a Sequential model.

ce_checkpoint_save called ce_module_param_count, which returned 0 for a
Sequential container: ce_sequential_add never exposed the children's
parameter/gradient tensors on the parent module.  Result:

  * save wrote a 20-byte header — no weights — and reported CE_OK;
  * load compared 0 == 0 and reported a successful load that restored
    nothing — silent weight loss behind a success status.

The fix registers each child's tensors on the container in
ce_sequential_add (list_append retains, so ownership stays safe).

Run: pytest tests/test_checkpoint_sequential.py -v
"""
import ctypes
import os
import sys

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cengine import ELU, Linear, Sequential, Sigmoid
from cengine.bindings import lib

# (8,6) + (8,) + (4,8) + (4,) = 92 float32 = 368 bytes of weights
HEADER_BYTES = 20  # CECK header observed for an empty parameter list
WEIGHT_BYTES = 92 * 4


def _build() -> Sequential:
    return Sequential(
        Linear(6, 8, name="ck_l1"), ELU(name="ck_elu"),
        Linear(8, 4, name="ck_l2"), Sigmoid(name="ck_sig"),
        name="ckpt_seq_model",
    )


def _snap(model: Sequential) -> list[np.ndarray]:
    return [
        np.array(
            np.ctypeslib.as_array(p.contents.data[:p.contents.numel]),
            copy=True,
        )
        for p in model.parameters()
    ]


def test_save_writes_actual_weights(tmp_path):
    """The checkpoint file must contain the weights, not just a header."""
    model = _build()
    path = str(tmp_path / "ck")

    status = lib.ce_checkpoint_save(
        path.encode("utf-8"), 0, ctypes.c_float(0.5), model._handle, None
    )
    assert status == 0, f"save failed with status {status}"

    bin_path = os.path.join(path, "checkpoint_epoch_0.bin")
    assert os.path.isfile(bin_path), f"missing checkpoint file: {bin_path}"
    size = os.path.getsize(bin_path)
    assert size >= HEADER_BYTES + WEIGHT_BYTES, (
        f"checkpoint is {size} bytes — weights (expected at least "
        f"{HEADER_BYTES + WEIGHT_BYTES} = header {HEADER_BYTES} + "
        f"{WEIGHT_BYTES} weight bytes) were not written: "
        f"ce_module_param_count returned 0 for the Sequential container"
    )
    model.close()


def test_load_restores_poisoned_weights(tmp_path):
    """A successful load must actually rewrite the model's parameters."""
    model = _build()
    path = str(tmp_path / "ck")

    original = _snap(model)
    status = lib.ce_checkpoint_save(
        path.encode("utf-8"), 0, ctypes.c_float(0.5), model._handle, None
    )
    assert status == 0, f"save failed with status {status}"

    for p in model.parameters():
        for i in range(p.contents.numel):
            p.contents.data[i] = -999.0

    status = lib.ce_checkpoint_load(path.encode("utf-8"), model._handle)
    assert status == 0, f"load failed with status {status}"

    restored = _snap(model)
    for idx, (a, b) in enumerate(zip(original, restored)):
        assert np.array_equal(a, b), (
            f"param {idx} was not restored by a CE_OK load — "
            f"maxdiff={np.abs(a - b).max():.4g} (poisoned values survive: "
            f"the checkpoint contained no weights)"
        )
    model.close()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
