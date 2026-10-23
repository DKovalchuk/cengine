"""
Regression tests for PyTorch checkpoint loading (Python-layer finding P8).

load_model_pytorch applied each parameter the moment its entry parsed,
skipping problems with a printed WARN and a `continue`:

  * a checkpoint missing keys loaded every OTHER parameter and left the
    rest at their initial values;
  * a size-mismatched tensor was skipped while its siblings loaded;

either way the model ended up holding a MIX of checkpoint and initial
weights with no error.  The loader is now two-phase: validate every
parameter (key present, size match, name/model count) first, raise
CEngineError listing all problems, and only then commit — all-or-nothing.

CI note: torch is not installed in CI — these tests self-skip there
(importorskip) and run wherever torch is available.

Run: pytest tests/test_pytorch_io.py -v
"""

import ctypes
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

torch = pytest.importorskip("torch")

from cengine.bindings import CEngineError, check, lib
from cengine.config import TrainingConfig
from cengine.module import Linear
from cengine.trainer import CEngineTrainer

check(lib.ce_init())


def _make_trainer() -> CEngineTrainer:
    model = Linear(19, 16, name="pt_model")
    cfg = TrainingConfig(total_epochs=1, batch_size=4, samples_per_epoch=4)
    return CEngineTrainer(model, cfg)


def _snapshot(params):
    return [
        [p.contents.data[i] for i in range(p.contents.numel)]
        for p in params
    ]


def _poison(params, value=-999.0):
    for p in params:
        for i in range(p.contents.numel):
            p.contents.data[i] = value


def _rewrite(path, mutate):
    """Load a .pt, let `mutate(state_dict)` alter it, save it back."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    mutate(sd)
    torch.save({"state_dict": sd, "metadata": {}}, path)


def test_pytorch_roundtrip(tmp_path):
    """Baseline: save → poison → load restores (pins the happy path)."""
    t = _make_trainer()
    params = t._model.parameters()
    original = _snapshot(params)

    path = str(tmp_path / "m.pt")
    t.save_model_pytorch(path)

    _poison(params)
    t.load_model_pytorch(path)
    assert _snapshot(params) == original, "roundtrip lost values"


def test_missing_key_is_rejected_atomically(tmp_path):
    """A checkpoint without one key must NOT load the others."""
    t = _make_trainer()
    params = t._model.parameters()

    path = str(tmp_path / "m.pt")
    t.save_model_pytorch(path)
    before = _snapshot(params)
    _rewrite(path, lambda sd: sd.pop("param_1", None))

    with pytest.raises(CEngineError):
        t.load_model_pytorch(path)

    after = _snapshot(params)
    assert after == before, (
        "model holds a mix of checkpoint and initial weights — params "
        "were applied before the missing key was rejected"
    )


def test_size_mismatch_is_rejected_atomically(tmp_path):
    """A size-mismatched tensor must NOT let siblings load."""
    t = _make_trainer()
    params = t._model.parameters()

    path = str(tmp_path / "m.pt")
    t.save_model_pytorch(path)
    before = _snapshot(params)

    def mutate(sd):
        sd["param_0"] = torch.zeros(params[0].contents.numel - 1)

    _rewrite(path, mutate)

    with pytest.raises(CEngineError):
        t.load_model_pytorch(path)

    after = _snapshot(params)
    assert after == before, (
        "model holds a mix of checkpoint and initial weights — the "
        "size-mismatched param was skipped while siblings were applied"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
