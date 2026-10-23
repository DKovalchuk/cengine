"""
Regression tests for ALU-generator lifecycle (Python-layer finding P5).

The generator returned by ce_alu_gen_create / ce_curriculum_make_generator
is a plain heap object, but Python never freed it: the per-epoch curriculum
stage change overwrote self._generator after only destroying the dataloader
(the dataloader does NOT own the generator), and _teardown destroyed neither
— one generator leaked per stage change plus one per trainer run.

There is now an explicit ce_alu_gen_destroy binding, used at both sites.
The leak itself is plain calloc (invisible to the memory tracker), so the
test pins the contract by (a) the binding existing and (b) a call counter
observing BOTH call sites during a real setup → train (stage change) →
teardown cycle.

Run: pytest tests/test_generator_lifecycle.py -v
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cengine.bindings import check, lib
from cengine.config import TrainingConfig
from cengine.module import Linear
from cengine.trainer import CEngineTrainer

check(lib.ce_init())


def test_destroy_binding_exists():
    assert hasattr(lib, "ce_alu_gen_destroy"), (
        "ce_alu_gen_destroy binding missing — Python has no way to free "
        "the ALU generator (leaks one per stage change and per run)"
    )


def test_trainer_frees_generators(monkeypatch):
    """Every generator the trainer creates must be destroyed exactly once."""
    real = lib.ce_alu_gen_destroy
    created: list = []
    destroyed: list = []

    def counting_create(*args):
        handle = real_create(*args)
        if handle:
            created.append(handle)
        return handle

    def counting_destroy(gen):
        destroyed.append(gen)
        return real(gen)

    # Wrap creation too, so failures to destroy SOME generator show up as
    # counts_mismatch, not just a zero counter.
    real_create = lib.ce_curriculum_make_generator
    monkeypatch.setattr(lib, "ce_curriculum_make_generator", counting_create)
    monkeypatch.setattr(lib, "ce_alu_gen_destroy", counting_destroy)

    model = Linear(19, 16, name="gen_lifecycle")
    # threshold huge + patience 1 → the curriculum advances after every
    # epoch, exercising the stage-change recreation path twice.
    cfg = TrainingConfig(
        total_epochs=2,
        batch_size=4,
        samples_per_epoch=8,
        loss_threshold=1e9,
        patience=1,
        seed=42,
    )
    trainer = CEngineTrainer(model, cfg)
    trainer._setup()
    trainer._run_loop()          # finally → _teardown

    # ≥2 destroys = both stage changes (epochs 0 and 1) plus teardown
    # (3 generators created, 3 destroyed with the fix).
    assert len(destroyed) >= 2, (
        f"ce_alu_gen_destroy called {len(destroyed)}x — stage-change and/or "
        "teardown paths leak the generator"
    )
    # No created generator may outlive the run (all must be destroyed).
    assert len(destroyed) == len(created), (
        f"generator leak: created={len(created)} destroyed={len(destroyed)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
