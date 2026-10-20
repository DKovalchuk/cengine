"""
cengine — Python bindings for the CEngine neural network training engine.

Provides a PyTorch-style API for defining models, configuring training, and
running high-performance training loops backed by the C engine's AVX2/SIMD
kernels, lock-free telemetry, and async data loading.

Quick start::

    from cengine.python import (
        CEngineTrainer, TrainingConfig,
        Sequential, Linear, ELU, Sigmoid,
    )

    model = Sequential(
        Linear(19, 128), ELU(),
        Linear(128, 128), ELU(),
        Linear(128, 16), Sigmoid(),
    )

    cfg = TrainingConfig(total_epochs=3000, learning_rate=1e-3)
    trainer = CEngineTrainer(model, cfg)
    trainer.start()  # non-blocking
    # ... monitor via trainer.get_telemetry()
    trainer.stop()
    trainer.join()
"""

from __future__ import annotations

# ── Core bindings (low-level) ──────────────────────────────────────────────
from .bindings import (
    CEngineError,
    CeAdam,
    CeAluGen,
    CeAluSample,
    CeBatch,
    CeCollector,
    CeComputeBackend,
    CeConfig,
    CeCurriculum,
    CeCurriculumStage,
    CeDataloader,
    CeLinear,
    CeLoadBalancer,
    CeMemoryStats,
    CeMetric,
    CeModule,
    CePerf,
    CeReporter,
    CeRingBuffer,
    CeSGD,
    CeScheduler,
    CeSequential,
    CeTape,
    CeTapeEntry,
    CeTask,
    CeTelemetrySnapshot,
    CeTensor,
    CeTensorList,
    check,
    lib,
)

# ── High-level API ─────────────────────────────────────────────────────────
from .config import TrainingConfig
from .module import (
    ELU,
    GumbelSoftmax,
    Linear,
    Module,
    RecurrentExpert,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
)
from .telemetry import MemoryStats, MetricPoint, TelemetryMonitor
from .trainer import CEngineTrainer

__all__ = [
    # High-level
    "CEngineTrainer",
    "TrainingConfig",
    "Module",
    "Linear",
    "ELU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "GumbelSoftmax",
    "Sequential",
    "TelemetryMonitor",
    "MetricPoint",
    "MemoryStats",
    # Error handling
    "CEngineError",
    "check",
    # Low-level bindings (for advanced usage)
    "lib",
    "CeConfig",
    "CeTensor",
    "CeTensorList",
    "CeModule",
    "CeLinear",
    "CeSequential",
    "CeAdam",
    "CeSGD",
    "CeScheduler",
    "CeMemoryStats",
    "CeMetric",
    "CeRingBuffer",
    "CeCollector",
    "CeReporter",
    "CeTelemetrySnapshot",
    "CePerf",
    "CeComputeBackend",
    "CeAluGen",
    "CeAluSample",
    "CeBatch",
    "CeCurriculum",
    "CeCurriculumStage",
    "CeDataloader",
    "CeTape",
    "CeTapeEntry",
    "CeTask",
    "CeLoadBalancer",
]
