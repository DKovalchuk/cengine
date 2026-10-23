"""
PyTorch bridge: architecture metadata and torch-compatible names.

``save_model_pytorch()`` wrote a state_dict, but its metadata carried no
architecture — only ``model_type``.  A .pt file therefore did not say HOW
to rebuild the equivalent torch module, so CUDA training required
hand-guessing the layers and their sizes.

MoE keys used ``expert.{i}`` (singular) while torch's ``nn.ModuleList``
(and the original KAN bridge, ``experts.{eid}.{pname}``) name them
``experts.{i}`` — the keys would not match on a load.

Run: pytest tests/test_pytorch_bridge.py -v
"""
import os
import sys

import pytest

# ── Path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cengine import (
    CEngineTrainer,
    ELU,
    Linear,
    Sequential,
    Sigmoid,
    TrainingConfig,
)
from cengine.bindings import c_int64, check, lib

check(lib.ce_init())


def _cfg(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        total_epochs=1,
        batch_size=4,
        samples_per_epoch=8,
        use_curriculum=False,
        min_bits=4,
        max_bits=4,
        checkpoint_dir=str(tmp_path / "ckpt"),
        seed=1,
    )


def test_metadata_contains_architecture(tmp_path):
    """The .pt metadata must describe the layers so torch can rebuild them."""
    torch = pytest.importorskip("torch")

    model = Sequential(
        Linear(6, 8, name="br_l1"), ELU(name="br_elu"),
        Linear(8, 4, name="br_l2"), Sigmoid(name="br_sig"),
        name="bridge_model",
    )
    trainer = CEngineTrainer(model, _cfg(tmp_path))
    path = str(tmp_path / "model.pt")
    trainer.save_model_pytorch(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "architecture" in ckpt["metadata"], (
        "metadata has no architecture — a .pt file does not say how to "
        "rebuild the equivalent torch module"
    )
    assert ckpt["metadata"]["architecture"] == [
        {"type": "linear", "in_features": 6, "out_features": 8},
        {"type": "elu", "alpha": 1.0},
        {"type": "linear", "in_features": 8, "out_features": 4},
        {"type": "sigmoid"},
    ]
    model.close()


class _FakeMoE:
    """Minimal stand-in exposing what _infer_param_names reads."""

    def __init__(self, n_experts: int) -> None:
        self.experts = [object() for _ in range(n_experts)]

    def parameters(self):
        return [None] * (len(self.experts) * 8)


def test_moe_param_names_match_torch_modulelist(tmp_path):
    """MoE keys must be experts.{i}.{name} like torch's nn.ModuleList."""
    trainer = CEngineTrainer(_FakeMoE(2), _cfg(tmp_path))
    names = trainer._infer_param_names()

    assert len(names) == 16
    assert all(n.startswith("experts.") for n in names), (
        f"keys must match torch nn.ModuleList naming (experts.{{i}}.name), "
        f"got: {names[:4]} ..."
    )
    assert names[0] == "experts.0.w1"


def test_sequential_state_dict_keys_match_torch(tmp_path):
    """Sequential checkpoints must use torch nn.Sequential numbering.

    A rebuilt torch.nn.Sequential numbers its children from 0 (activations
    included), so its state_dict keys are ``0.weight``/``2.weight`` — the
    cengine state_dict must use the same keys for load_state_dict(strict).
    """
    torch = pytest.importorskip("torch")

    model = Sequential(
        Linear(6, 8, name="k_l1"), ELU(name="k_elu"),
        Linear(8, 4, name="k_l2"), Sigmoid(name="k_sig"),
        name="keys_model",
    )
    trainer = CEngineTrainer(model, _cfg(tmp_path))
    path = str(tmp_path / "keys.pt")
    trainer.save_model_pytorch(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = set(ckpt["state_dict"].keys())
    assert keys == {"0.weight", "0.bias", "2.weight", "2.bias"}, (
        f"state_dict keys must follow torch nn.Sequential numbering, got {sorted(keys)}"
    )
    model.close()


def test_torch_model_matches_cengine_forward(tmp_path):
    """The rebuilt torch module must reproduce the cengine forward pass."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from cengine.pytorch_bridge import load_model_torch

    model = Sequential(
        Linear(6, 8, name="f_l1"), ELU(name="f_elu"),
        Linear(8, 4, name="f_l2"), Sigmoid(name="f_sig"),
        name="fwd_model",
    )
    trainer = CEngineTrainer(model, _cfg(tmp_path))
    path = str(tmp_path / "fwd.pt")
    trainer.save_model_pytorch(path)

    torch_model = load_model_torch(path)
    torch_model.eval()

    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 6)).astype(np.float32)

    # Same input through both engines.
    in_t = lib.ce_tensor_zeros((c_int64 * 2)(4, 6), 2)
    for i, v in enumerate(x.ravel()):
        in_t.contents.data[i] = float(v)
    out_t = lib.ce_tensor_zeros((c_int64 * 2)(4, 4), 2)
    model.forward(in_t, out_t)
    y_cengine = np.ctypeslib.as_array(out_t.contents.data[:16]).reshape(4, 4).copy()

    with torch.no_grad():
        y_torch = torch_model(torch.from_numpy(x)).numpy()

    assert np.allclose(y_cengine, y_torch, atol=1e-5), (
        f"torch module disagrees with cengine: maxdiff="
        f"{np.abs(y_cengine - y_torch).max():.3g}"
    )
    model.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
