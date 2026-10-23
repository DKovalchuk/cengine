"""
Regression tests for TrainingConfig validation (Python-layer finding P7).

TrainingConfig accepted anything: a loss_type typo ("BCE", "Bce", ...)
silently fell into the trainer's `else` branch and trained with MSE
instead of the requested BCE, min_bits > max_bits produced an
impossible generator range, and zero/negative batch or epoch counts
reached the C layer unchecked (the C trainer rejects them — the Python
config should too, at construction time).

Run: pytest tests/test_config_validation.py -v
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cengine.config import TrainingConfig


def test_default_config_still_constructs():
    """The shipped defaults must remain valid (pin)."""
    cfg = TrainingConfig()
    assert cfg.loss_type == "bce"
    assert cfg.optimizer == "adamw"
    assert cfg.min_bits == 2 and cfg.max_bits == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loss_type": "BCE"},                 # typo → silently trained MSE
        {"loss_type": "bce_with_logits"},     # unknown → silently MSE
        {"optimizer": "ADAM"},                # unknown → fails only at setup
        {"min_bits": 5, "max_bits": 3},       # impossible generator range
        {"min_bits": 0, "max_bits": 3},       # min_bits below 1
        {"batch_size": 0},                    # reaches the C layer unchecked
        {"total_epochs": 0},                  # training can never run
    ],
    ids=["loss-typo", "loss-unknown", "optimizer", "bits-order",
         "bits-min", "batch", "epochs"],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
