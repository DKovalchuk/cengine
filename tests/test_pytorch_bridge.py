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
from cengine.bindings import check, lib

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
