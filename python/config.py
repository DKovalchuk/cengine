"""
Pythonic training configuration that maps cleanly to C engine structs.

Usage::

    cfg = TrainingConfig(learning_rate=1e-3, total_epochs=3000)
    trainer = CEngineTrainer(model, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TrainingConfig"]


@dataclass
class TrainingConfig:
    """High-level training hyperparameters.

    Every field has a sensible default that works for the ALU task out of the
    box.  The ``to_c_params()`` method converts this into a dict of primitive
    values that the ``CEngineTrainer`` uses to construct individual C objects
    (optimizer, scheduler, dataloader, etc.).
    """

    # ── Model ──────────────────────────────────────────────────────────────
    batch_size: int = 256
    samples_per_epoch: int = 2048
    total_epochs: int = 6000

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer: str = "adamw"  # 'adamw' | 'sgd'
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    momentum: float = 0.9  # SGD only

    # ── Loss ───────────────────────────────────────────────────────────────
    loss_type: str = "bce"  # 'bce' | 'mse'
    lambda_balance: float = 0.1
    lambda_z: float = 0.001
    lambda_carry: float = 0.3

    # ── Temperature annealing ──────────────────────────────────────────────
    temp_start: float = 2.0
    temp_end: float = 0.5
    temp_anneal_epochs: int = 2000

    # ── Scheduler ──────────────────────────────────────────────────────────
    pct_start: float = 0.1
    div_factor: float = 25.0
    final_div_factor: float = 1e4

    # ── Gradient clipping ──────────────────────────────────────────────────
    grad_clip_norm: float = 1.0

    # ── Curriculum ─────────────────────────────────────────────────────────
    use_curriculum: bool = True
    loss_threshold: float = 0.015
    patience: int = 5
    max_epochs_per_stage: int = 2000

    # ── Threading ──────────────────────────────────────────────────────────
    num_workers: int = 0  # 0 = auto-detect

    # ── Telemetry ──────────────────────────────────────────────────────────
    enable_telemetry: bool = True
    telemetry_interval: int = 1

    # ── Debug logging ──────────────────────────────────────────────────────
    enable_debug_logging: bool = False

    # ── Checkpointing ──────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 100

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── ALU data specifics ─────────────────────────────────────────────────
    min_bits: int = 2
    max_bits: int = 8
    ops: list[int] = field(default_factory=lambda: [0, 1, 2, 3])  # ADD, SUB, AND, XOR

    # ── Model dimensions (override auto-detection) ───────────────────────
    model_input_dim: int = 0   # 0 = auto-detect from first expert layer
    model_output_dim: int = 0  # 0 = auto-detect from last layer

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def __post_init__(self) -> None:
        # A loss_type typo ("BCE") would fall through the trainer's
        # `else` branch and silently train with MSE instead.
        if self.loss_type not in ("bce", "mse"):
            raise ValueError(
                f"loss_type must be 'bce' or 'mse', got {self.loss_type!r}"
            )
        if self.optimizer not in ("adamw", "sgd"):
            raise ValueError(
                f"optimizer must be 'adamw' or 'sgd', got {self.optimizer!r}"
            )
        if self.min_bits < 1 or self.max_bits < self.min_bits:
            raise ValueError(
                f"bit range must satisfy 1 <= min_bits <= max_bits, "
                f"got {self.min_bits}..{self.max_bits}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {self.total_epochs}")

    # -----------------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------------

    @property
    def max_lr(self) -> float:
        """Peak LR for the OneCycle scheduler."""
        return self.learning_rate

    @property
    def min_lr(self) -> float:
        """Minimum LR for the OneCycle scheduler."""
        return self.learning_rate / self.div_factor

    def temperature_at(self, epoch: int) -> float:
        """Linear temperature annealing from temp_start → temp_end."""
        if epoch >= self.temp_anneal_epochs:
            return self.temp_end
        t = epoch / max(self.temp_anneal_epochs, 1)
        return self.temp_start + t * (self.temp_end - self.temp_start)

    def to_c_params(self) -> dict[str, Any]:
        """Return a flat dict of all parameters for the trainer."""
        return {
            "batch_size": self.batch_size,
            "samples_per_epoch": self.samples_per_epoch,
            "total_epochs": self.total_epochs,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "momentum": self.momentum,
            "loss_type": self.loss_type,
            "lambda_balance": self.lambda_balance,
            "lambda_z": self.lambda_z,
            "lambda_carry": self.lambda_carry,
            "temp_start": self.temp_start,
            "temp_end": self.temp_end,
            "temp_anneal_epochs": self.temp_anneal_epochs,
            "pct_start": self.pct_start,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
            "grad_clip_norm": self.grad_clip_norm,
            "use_curriculum": self.use_curriculum,
            "loss_threshold": self.loss_threshold,
            "patience": self.patience,
            "max_epochs_per_stage": self.max_epochs_per_stage,
            "num_workers": self.num_workers,
            "enable_telemetry": self.enable_telemetry,
            "enable_debug_logging": self.enable_debug_logging,
            "telemetry_interval": self.telemetry_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "save_every_n_epochs": self.save_every_n_epochs,
            "seed": self.seed,
            "min_bits": self.min_bits,
            "max_bits": self.max_bits,
            "ops": self.ops,
        }

    def __repr__(self) -> str:
        return (
            f"TrainingConfig(optimizer={self.optimizer!r}, lr={self.learning_rate}, "
            f"epochs={self.total_epochs}, batch={self.batch_size}, "
            f"loss={self.loss_type!r}, bits={self.min_bits}-{self.max_bits})"
        )
