"""
Regression tests for the NaN skip-paths (Python-layer findings P3/P4).

_run_loop() starts every epoch with ce_collector_start_epoch() and pairs it
with ce_collector_end_epoch() on the normal path — but both NaN escape
paths skipped the pair:

  * P3 — NaN loss: released the tensors and `break`-ed the training loop
    with the message "skipping rest of epoch" (it actually STOPS the whole
    run) and never emitted the epoch's EPOCH_TIME_MS record;
  * P4 — NaN gradient: `continue`-d to the next epoch (correctly skipping
    the optimizer step) but also never emitted the epoch record.

Telemetry consumers therefore saw started epochs silently vanish on every
NaN skip, and P3's message contradicted its behaviour.

Behaviour choices kept: NaN loss stops training (defensive — weights have
diverged), NaN grad skips just that epoch's step.  Both paths now close
the epoch pair, and P3's message says what the code does.

Run: pytest tests/test_nan_paths.py -v
"""

import math
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python.bindings import check, lib
from python.config import TrainingConfig
from python.module import Linear
from python.trainer import CEngineTrainer

check(lib.ce_init())


class _NanGradLinear(Linear):
    """Linear whose backward poisons the weight gradient with NaN."""

    def backward(self, grad_output, grad_input):
        super().backward(grad_output, grad_input)
        g = self.gradients()[0]
        g.contents.data[0] = float("nan")


def _run_and_drain(model, total_epochs: int, poison: bool):
    """Run the loop synchronously with a ring reference kept alive.

    Returns (epoch_time_records, final_epoch_index, captured_caller).
    The ring buffer object is Python-owned; we hold a reference so
    poll() stays valid after the loop's teardown released the trainer's.
    """
    cfg = TrainingConfig(
        total_epochs=total_epochs,
        batch_size=4,
        samples_per_epoch=8,
        seed=42,
    )
    trainer = CEngineTrainer(model, cfg)
    if poison:
        for p in trainer._model.parameters():
            p.contents.data[0] = float("nan")

    trainer._setup()
    ring_ref = trainer._ring_buffer_obj        # keep the struct alive
    monitor = trainer._telemetry
    assert ring_ref is not None and monitor is not None

    trainer._run_loop()
    assert not trainer.is_running

    points = monitor.poll()
    epoch_times = [p for p in points if p.name == "epoch_time_ms"]
    return epoch_times, trainer._current_epoch


def test_nan_loss_stops_training_and_closes_epoch(capsys):
    """P3: the break must emit the epoch metric and say it stops."""
    times, final_epoch = _run_and_drain(
        Linear(19, 16, name="nan_loss_model"), total_epochs=3, poison=True,
    )
    out = capsys.readouterr().out

    # Behaviour kept: training stopped at the first (diverged) epoch.
    assert final_epoch == 0, "NaN loss must stop the run at epoch 0"
    assert "stopping training" in out, (
        f"message must state the actual behaviour; got: {out.strip()!r}"
    )

    # The started epoch must be closed: one EPOCH_TIME_MS record.
    assert len(times) == 1, (
        f"expected 1 epoch_time_ms record for the NaN-loss epoch, got "
        f"{len(times)} — ce_collector_end_epoch was skipped on the break "
        "path (start/end pair corrupted)"
    )


def test_nan_grad_skips_step_but_keeps_epoch_pair():
    """P4: the continue must still close the epoch's telemetry pair."""
    times, final_epoch = _run_and_drain(
        _NanGradLinear(19, 16, name="nan_grad_model"), total_epochs=3,
        poison=False,
    )

    # Behaviour kept: the loop ran every epoch (only steps were skipped).
    assert final_epoch == 2, "NaN grad must skip the step, not the epochs"
    assert len(times) == 3, (
        f"expected 3 epoch_time_ms records (one per epoch), got "
        f"{len(times)} — ce_collector_end_epoch was skipped on the "
        "NaN-grad continue path"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
