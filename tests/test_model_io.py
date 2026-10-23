"""
Regression tests for the Python raw-checkpoint loader (P2).

load_model()'s CEPY fallback used the file's own metadata to decide how
much to write into the model's buffers:

  - ndim/shape read from the file were NEVER compared with the model → a
    checkpoint whose parameters were stored with a different layout was
    silently loaded (transposed data = corrupted weights);
  - numel came from the file and fed ctypes.memmove straight into the
    model-sized tensor → oversized numel = heap OOB write, and a
    truncated file produced a SHORT raw buffer that memmove reads past
    (silent garbage = mixed weights, no rollback).

The loader must now validate EVERY parameter (ndim, shape, numel, byte
length) against the model BEFORE touching any tensor — all-or-nothing,
like the C-side checkpoint load (finding B2).

Run: pytest tests/test_model_io.py -v
"""

import ctypes
import os
import struct
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python.bindings import CE_OK, CEngineError, check, lib
from python.module import Linear
from python.trainer import CEngineTrainer
from python.config import TrainingConfig

check(lib.ce_init())


def _make_trainer() -> CEngineTrainer:
    model = Linear(4, 3, name="io_model")
    cfg = TrainingConfig(total_epochs=1, batch_size=4, samples_per_epoch=4)
    return CEngineTrainer(model, cfg)


def _pack_cepy(params, shape_override=None, truncate_last=0) -> bytes:
    """Serialise params in the CEPY fallback format.

    shape_override: optional {param_index: new_shape} to craft a
    layout-mismatched file (same numel, different ndim/shape).
    truncate_last: cut this many bytes off the end (craft a short file).
    """
    out = bytearray()
    out += b"CEPY"
    out += struct.pack("<I", 1)                       # version
    out += struct.pack("<I", len(params))             # n_params
    for i, p in enumerate(params):
        ndim = p.contents.ndim
        shape = [p.contents.shape[d] for d in range(ndim)]
        if shape_override and i in shape_override:
            shape = list(shape_override[i])
            ndim = len(shape)
        numel = p.contents.numel
        out += struct.pack("<I", ndim)
        for s in shape:
            out += struct.pack("<I", s)
        out += struct.pack("<Q", numel)
        addr = ctypes.addressof(p.contents.data.contents)
        out += ctypes.string_at(addr, numel * 4)
    if truncate_last:
        out = out[:-truncate_last]
    return bytes(out)


def _write(path: str, blob: bytes) -> None:
    with open(path, "wb") as f:
        f.write(blob)


def _snapshot(params):
    return [
        (p.contents.numel,
         [p.contents.data[i] for i in range(p.contents.numel)])
        for p in params
    ]


def test_valid_cepy_roundtrip_loads(tmp_path):
    """A well-formed CEPY file must load (baseline for the rejection tests)."""
    t = _make_trainer()
    params = t._model.parameters()
    original = _snapshot(params)          # real initial values
    blob = _pack_cepy(params)             # file carries those values

    # Poison the model so the load is observable
    for p in params:
        for i in range(p.contents.numel):
            p.contents.data[i] = -999.0

    _write(str(tmp_path / "checkpoint_latest.bin"), blob)
    t.load_model(str(tmp_path))                       # must not raise

    assert _snapshot(params) == original, (
        "valid CEPY checkpoint must restore the saved values"
    )


def test_shape_mismatch_is_rejected_without_touching_params(tmp_path):
    """Same numel, different layout → reject; params must stay untouched."""
    t = _make_trainer()
    params = t._model.parameters()
    # weight is (3,4): craft the file storing it as (4,3) — same numel
    before = _snapshot(params)

    blob = _pack_cepy(params, shape_override={0: [4, 3]})
    _write(str(tmp_path / "checkpoint_latest.bin"), blob)

    with pytest.raises(CEngineError):
        t.load_model(str(tmp_path))

    after = _snapshot(params)
    assert after == before, (
        "params were partially rewritten by a layout-mismatched checkpoint "
        "(no rollback — mixed weights)"
    )


def test_truncated_file_is_rejected_without_touching_params(tmp_path):
    """A short file must not partially load (silent garbage before)."""
    t = _make_trainer()
    params = t._model.parameters()
    before = _snapshot(params)

    # Cut into the LAST parameter's data section (bias = 3 floats = 12 B)
    blob = _pack_cepy(params, truncate_last=6)
    _write(str(tmp_path / "checkpoint_latest.bin"), blob)

    with pytest.raises(CEngineError):
        t.load_model(str(tmp_path))

    after = _snapshot(params)
    assert after == before, (
        "params were partially rewritten by a truncated checkpoint "
        "(no rollback — mixed weights)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
