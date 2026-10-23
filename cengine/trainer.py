"""
High-level training orchestrator.

``CEngineTrainer`` wires together a Python ``Module``, a ``TrainingConfig``,
and the C engine's optimizer / scheduler / dataloader / telemetry subsystems.
It runs the training loop in a background thread so the caller retains control
(observe telemetry, request a stop, etc.).

Usage::

    model = Sequential(
        Linear(19, 128), ELU(),
        Linear(128, 128), ELU(),
        Linear(128, 16), Sigmoid(),
    )
    cfg = TrainingConfig(total_epochs=3000, learning_rate=1e-3)
    trainer = CEngineTrainer(model, cfg)
    trainer.start()          # non-blocking (background thread)
    # ...
    trainer.stop()           # request graceful stop
    trainer.join()           # wait for thread to finish
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
import time
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_int64, c_uint64, sizeof
from typing import Any, Callable

from .bindings import (
    CE_FLOAT32,
    CE_OK,
    CeAluGen,
    CeAluSample,
    CeBatch,
    CeCollector,
    CeConfig,
    CeCurriculum,
    CeDataloader,
    CeMemoryStats,
    CeMetric,
    CeRingBuffer,
    CeScheduler,
    CeTensor,
    CEngineError,
    check,
    lib,
)
from .config import TrainingConfig
from .module import Module
from .telemetry import MemoryStats, TelemetryMonitor

__all__ = ["CEngineTrainer"]

# Optional: TrainingState for TUI channel
try:
    from ..src.training_state import TrainingState  # type: ignore
except ImportError:
    try:
        from training_state import TrainingState  # type: ignore
    except ImportError:
        TrainingState = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensor_array(tensors: list[POINTER(CeTensor)]):
    """Create a ``ce_tensor_t**`` (C array of pointers) from a Python list."""
    arr_type = POINTER(CeTensor) * len(tensors)
    return arr_type(*tensors)


def _copy_sample_data(
    dst_tensor: POINTER(CeTensor),
    src_tensor: POINTER(CeTensor),
    offset: int,
) -> None:
    """Copy *src_tensor*'s float data into *dst_tensor* starting at element *offset*."""
    n = src_tensor.contents.numel
    src_addr = ctypes.addressof(src_tensor.contents.data.contents)
    dst_addr = ctypes.addressof(dst_tensor.contents.data.contents) + offset * sizeof(c_float)
    ctypes.memmove(dst_addr, src_addr, n * sizeof(c_float))


def _batch_to_tensors(
    batch_ptr: POINTER(CeBatch),
    max_input_dim: int = 0,
    max_target_dim: int = 0,
    recurrent: bool = False,
) -> tuple[POINTER(CeTensor), POINTER(CeTensor), int, POINTER(CeTensor)]:
    """Convert a C batch into (input, target, n_bits, mask).

    When ``max_input_dim`` / ``max_target_dim`` (the model's widths) are
    given, rows are model-anchored exactly like C's
    ``ce_trainer_prepare_batch``:

        input  = [a_bits(0..n), b_bits(anchor..anchor+n),
                  op_code(2*anchor..+3)]   anchor = (max_input_dim-3)//2
        target = [result(0..n), carry(t_anchor..t_anchor+n)]
                                        t_anchor = max_target_dim//2

    Without model dims the legacy batch anchor is kept: anchor = n_bits
    (max over the batch), dim = 2n+3 (input) / 2n (target).

    Mask is 1.0 at valid result/carry positions (model-anchored when the
    model dims are known), 0.0 at padding.  Variable n_bits across samples
    is handled by zero-padding shorter samples.
    """
    batch = batch_ptr.contents
    n_samples = batch.count
    if n_samples == 0:
        raise CEngineError("Empty batch from dataloader")

    # Find max n_bits across all samples (for consistent tensor dimensions)
    n_bits = max(batch.samples[i].contents.n_bits for i in range(n_samples))

    if recurrent:
        # Recurrent layout: [batch, T, 6] per sample
        # 6 = [a_bit, b_bit, carry_placeholder(0), op_0, op_1, op_2]
        T = n_bits  # max sequence length
        in_shape = (c_int64 * 3)(n_samples, T, 6)
        tgt_shape = (c_int64 * 3)(n_samples, T, 2)  # [result_bit, carry_bit]
        mask_shape = (c_int64 * 3)(n_samples, T, 2)

        input_tensor = lib.ce_tensor_zeros(in_shape, 3)
        target_tensor = lib.ce_tensor_zeros(tgt_shape, 3)
        mask_tensor = lib.ce_tensor_zeros(mask_shape, 3)

        for i in range(n_samples):
            sample = batch.samples[i].contents
            sn = sample.n_bits
            for t in range(sn):
                base_in = (i * T + t) * 6
                base_tgt = (i * T + t) * 2
                input_tensor.contents.data[base_in + 0] = sample.a_bits.contents.data[t]
                input_tensor.contents.data[base_in + 1] = sample.b_bits.contents.data[t]
                input_tensor.contents.data[base_in + 2] = 0.0  # carry placeholder
                input_tensor.contents.data[base_in + 3] = sample.op_code.contents.data[0]
                input_tensor.contents.data[base_in + 4] = sample.op_code.contents.data[1]
                input_tensor.contents.data[base_in + 5] = sample.op_code.contents.data[2]
                target_tensor.contents.data[base_tgt + 0] = sample.result_bits.contents.data[t]
                target_tensor.contents.data[base_tgt + 1] = sample.carry_bits.contents.data[t]
                # mask = 1.0 for valid timesteps
                mask_tensor.contents.data[base_tgt + 0] = 1.0
                mask_tensor.contents.data[base_tgt + 1] = 1.0

        return input_tensor, target_tensor, n_bits, mask_tensor

    # Flat layout — rows are anchored on the MODEL's dimensions when known,
    # mirroring ce_trainer_prepare_batch in C (final = (input_dim - 3) / 2):
    #   input  = [a(0..n), b(anchor..anchor+n), op_code(2*anchor..+3)]
    #   target = [result(0..n), carry(t_anchor..t_anchor+n)]
    # Anchoring on the BATCH's max n_bits (previous behaviour) shifted
    # b/op/carry/mask into columns the model reads as different features
    # whenever the batch was narrower than the model — op_code landed in
    # the b_bits columns, loss compared result slots against carry.
    # op_code is a 3-float binary encoding — the row must reserve 3 slots,
    # otherwise the copy below writes one float past the row (heap OOB write).
    if max_input_dim > 0 and max_input_dim >= 2 * n_bits + 3:
        input_dim = max_input_dim
        in_anchor = (max_input_dim - 3) // 2
    else:
        # No usable model dims (or the model is narrower than this batch —
        # that mismatch is reported loudly by the C matmul): fall back to
        # the legacy batch-n_bits anchor.
        input_dim = 2 * n_bits + 3
        in_anchor = n_bits
    if max_target_dim > 0 and max_target_dim >= 2 * n_bits:
        target_dim = max_target_dim
        tgt_anchor = max_target_dim // 2
    else:
        target_dim = 2 * n_bits
        tgt_anchor = n_bits

    # Allocate batch tensors (zero-initialised so shorter samples are padded)
    in_shape = (c_int64 * 2)(n_samples, input_dim)
    tgt_shape = (c_int64 * 2)(n_samples, target_dim)
    mask_shape = (c_int64 * 2)(n_samples, target_dim)

    input_tensor = lib.ce_tensor_zeros(in_shape, 2)
    target_tensor = lib.ce_tensor_zeros(tgt_shape, 2)
    mask_tensor = lib.ce_tensor_zeros(mask_shape, 2)
    if not input_tensor or not target_tensor:
        raise CEngineError("Failed to allocate batch tensors")

    # Fill data
    for i in range(n_samples):
        sample = batch.samples[i].contents
        sn = sample.n_bits  # this sample's actual bit width
        row_offset = i * input_dim

        # a_bits → input[i, 0:sn]
        _copy_sample_data(input_tensor, sample.a_bits, row_offset)
        # b_bits → input[i, anchor:anchor+sn]
        _copy_sample_data(input_tensor, sample.b_bits, row_offset + in_anchor)
        # op_code → input[i, 2*anchor:2*anchor+3]
        _copy_sample_data(input_tensor, sample.op_code, row_offset + 2 * in_anchor)

        tgt_offset = i * target_dim
        # result_bits → target[i, 0:sn]
        _copy_sample_data(target_tensor, sample.result_bits, tgt_offset)
        # carry_bits → target[i, t_anchor:t_anchor+sn]
        _copy_sample_data(target_tensor, sample.carry_bits, tgt_offset + tgt_anchor)

        # mask = 1.0 for valid positions (result and carry, model-anchored)
        for t in range(sn):
            mask_tensor.contents.data[i * target_dim + t] = 1.0
            mask_tensor.contents.data[i * target_dim + tgt_anchor + t] = 1.0

    return input_tensor, target_tensor, n_bits, mask_tensor


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CEngineTrainer:
    """Full training orchestrator.

    Args:
        model: A built ``Module`` (e.g. ``Sequential(…)``).
        config: Training hyperparameters.
        on_epoch_end: Optional callback ``(epoch, loss, trainer) -> None``.
    """

    def __init__(
        self,
        model: Module,
        config: TrainingConfig,
        on_epoch_end: Callable[[int, float, CEngineTrainer], None] | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._on_epoch_end = on_epoch_end

        # C objects (created in _setup)
        self._optimizer = None
        self._scheduler = None
        self._curriculum = None
        self._dataloader = None
        self._ring_buffer = None
        self._collector = None
        self._reporter = None
        self._telemetry: TelemetryMonitor | None = None

        # State
        self._current_epoch = 0
        self._current_loss = float("inf")
        self._best_loss = float("inf")
        self._running = False
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._training_state = None  # TUI channel

        # Ensure C engine is initialised
        check(lib.ce_init())

        # Pass debug logging flag to C engine
        cfg = config
        c_config = CeConfig()
        c_config.num_threads = cfg.num_workers if cfg.num_workers > 0 else 0
        c_config.enable_telemetry = cfg.enable_telemetry
        c_config.enable_perf_counters = False
        c_config.enable_debug_logging = cfg.enable_debug_logging
        c_config.memory_pool_size = 0  # auto
        check(lib.ce_configure(ctypes.byref(c_config)))

    # ── Public API ────────────────────────────────────────────────────────

    def start(self, blocking: bool = False) -> None:
        """Start training.

        Args:
            blocking: If ``True``, block the calling thread until training
                finishes.  If ``False`` (default), run in a background thread.
        """
        if self._running:
            raise CEngineError("Training is already running")

        self._stop_requested = False
        self._setup()

        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Request a graceful stop after the current epoch finishes."""
        self._stop_requested = True

    def join(self, timeout: float | None = None) -> None:
        """Wait for the training thread to finish."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def current_loss(self) -> float:
        return self._current_loss

    @property
    def best_loss(self) -> float:
        return self._best_loss

    def get_telemetry(self) -> TelemetryMonitor:
        """Return the telemetry monitor (available after ``start``)."""
        if self._telemetry is None:
            raise CEngineError("Telemetry not initialised — call start() first")
        return self._telemetry

    def get_memory_stats(self) -> MemoryStats:
        """Return current memory statistics from the C tracker."""
        c_stats = lib.ce_tracker_get_stats()
        return MemoryStats.from_c(c_stats)

    def set_extra_metrics(
        self,
        *,
        per_op_accuracy: dict[str, float] | None = None,
        per_bit_accuracy: dict[str, float] | None = None,
    ) -> None:
        """Pass extra metrics (accuracy, etc.) to the training state channel.

        Call this from the on_epoch_end callback to enrich the TUI dashboard.
        """
        if self._training_state is None:
            return
        try:
            self._training_state._last_per_op = per_op_accuracy or {}
            self._training_state._last_per_bit = per_bit_accuracy or {}
        except Exception:
            pass

    # ── Setup / teardown ──────────────────────────────────────────────────

    def _setup(self) -> None:
        """Create all C training objects."""
        cfg = self._config

        # ── Telemetry ─────────────────────────────────────────────────────
        # Allocate the ring buffer as a local C struct (196 KB).
        # Keep a strong reference so the struct isn't garbage-collected.
        self._ring_buffer_obj = CeRingBuffer()
        self._ring_buffer = ctypes.pointer(self._ring_buffer_obj)
        self._collector = lib.ce_collector_create(self._ring_buffer)
        if not self._collector:
            raise CEngineError("ce_collector_create returned NULL")
        self._reporter = lib.ce_reporter_create(self._ring_buffer)
        if not self._reporter:
            raise CEngineError("ce_reporter_create returned NULL")
        self._telemetry = TelemetryMonitor(self._ring_buffer, self._reporter)

        # ── Curriculum ────────────────────────────────────────────────────
        if cfg.use_curriculum:
            self._curriculum = lib.ce_curriculum_create(
                c_float(cfg.loss_threshold),
                c_int(cfg.patience),
                c_int(cfg.max_epochs_per_stage),
                c_int(cfg.max_bits),
            )
            if not self._curriculum:
                raise CEngineError("ce_curriculum_create returned NULL")
            self._generator = lib.ce_curriculum_make_generator(
                self._curriculum, c_int64(cfg.seed),
            )
        else:
            # No curriculum: generator covers all ops and full bit range from start
            self._curriculum = None
            ops_arr = (c_int * 4)(0, 1, 2, 3)  # ADD, SUB, AND, XOR
            self._generator = lib.ce_alu_gen_create(
                c_int(cfg.min_bits), c_int(cfg.max_bits),
                ops_arr, c_int(4), c_int64(cfg.seed),
            )
        if not self._generator:
            raise CEngineError("Generator creation returned NULL")

        self._dataloader = lib.ce_dataloader_create(
            self._generator,
            c_int(cfg.batch_size),
            c_int(cfg.samples_per_epoch),
            c_int64(cfg.seed),
        )
        if not self._dataloader:
            raise CEngineError("ce_dataloader_create returned NULL")

        # ── Optimizer ─────────────────────────────────────────────────────
        params = self._model.parameters()
        grads = self._model.gradients()
        n_params = len(params)

        if n_params == 0:
            raise CEngineError("Model has no parameters")

        params_arr = _make_tensor_array(params)
        grads_arr = _make_tensor_array(grads)

        if cfg.optimizer == "adamw":
            self._optimizer = lib.ce_adam_create(
                c_float(cfg.learning_rate),
                c_float(cfg.beta1),
                c_float(cfg.beta2),
                c_float(cfg.eps),
                c_float(cfg.weight_decay),
                params_arr,
                grads_arr,
                c_int(n_params),
            )
            self._opt_type = "adam"
        elif cfg.optimizer == "sgd":
            self._optimizer = lib.ce_sgd_create(
                c_float(cfg.learning_rate),
                c_float(cfg.momentum),
                c_float(cfg.weight_decay),
                params_arr,
                grads_arr,
                c_int(n_params),
            )
            self._opt_type = "sgd"
        else:
            raise CEngineError(f"Unknown optimizer: {cfg.optimizer!r}")

        if not self._optimizer:
            raise CEngineError("Optimizer creation returned NULL")

        # ── Scheduler ─────────────────────────────────────────────────────
        total_steps = cfg.total_epochs * max(cfg.samples_per_epoch // cfg.batch_size, 1)
        self._scheduler = lib.ce_scheduler_create(
            c_float(cfg.max_lr),
            c_float(cfg.pct_start),
            c_float(cfg.div_factor),
            c_float(cfg.final_div_factor),
            c_int(total_steps),
        )
        if not self._scheduler:
            raise CEngineError("ce_scheduler_create returned NULL")

        # ── Gradient pointers (cached for clipping) ───────────────────────
        self._grads_arr = grads_arr
        self._n_grads = len(grads)
        self._params_arr = params_arr  # prevent GC
        self._params_list = params     # prevent GC of tensor pointers
        self._grads_list = grads       # prevent GC of gradient pointers

        # ── Training state channel (for TUI) ─────────────────────────────
        if TrainingState is not None:
            try:
                self._training_state = TrainingState()
                self._training_state.update(
                    epoch=0, total_epochs=cfg.total_epochs,
                    loss=float("inf"), best_loss=float("inf"),
                    config=cfg.to_c_params(), status="training",
                )
            except Exception:
                self._training_state = None

        # ── Detect model dimensions ──────────────────────────────────────
        # Use config overrides if provided, otherwise auto-detect
        self._out_dim = cfg.model_output_dim if cfg.model_output_dim > 0 else None
        self._in_dim = cfg.model_input_dim if cfg.model_input_dim > 0 else None

        if self._out_dim is None:
            for p in reversed(params):
                if p.contents.ndim == 2:
                    self._out_dim = p.contents.shape[0]
                    break
        if self._in_dim is None:
            for p in params:
                if p.contents.ndim == 2:
                    in_d = p.contents.shape[1]
                    if in_d > 16:  # skip router layers (small input)
                        self._in_dim = in_d
                        break
            if self._in_dim is None:
                for p in params:
                    if p.contents.ndim == 2:
                        self._in_dim = p.contents.shape[1]
                        break

    def _teardown(self) -> None:
        """Destroy all C objects."""
        # Close training state channel
        if self._training_state is not None:
            try:
                self._training_state.close(status="stopped")
            except Exception:
                pass
            self._training_state = None

        if self._dataloader:
            lib.ce_dataloader_destroy(self._dataloader)
            self._dataloader = None
        if self._generator:
            # Freed after the dataloader (which holds a borrowed pointer).
            lib.ce_alu_gen_destroy(self._generator)
            self._generator = None
        if self._curriculum:
            lib.ce_curriculum_destroy(self._curriculum)
            self._curriculum = None
        if self._optimizer:
            if self._opt_type == "adam":
                lib.ce_adam_destroy(self._optimizer)
            else:
                lib.ce_sgd_destroy(self._optimizer)
            self._optimizer = None
        if self._scheduler:
            lib.ce_scheduler_destroy(self._scheduler)
            self._scheduler = None
        if self._reporter:
            lib.ce_reporter_destroy(self._reporter)
            self._reporter = None
        if self._collector:
            lib.ce_collector_destroy(self._collector)
            self._collector = None
        # ring_buffer is Python-managed via ctypes — no C destroy needed
        self._ring_buffer = None
        self._ring_buffer_obj = None

    # ── Training loop ─────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main training loop (runs in caller or background thread)."""
        self._running = True
        cfg = self._config
        model = self._model

        try:
            for epoch in range(cfg.total_epochs):
                if self._stop_requested:
                    break

                self._current_epoch = epoch
                temperature = cfg.temperature_at(epoch)

                # Pass temperature to model for router softmax scaling
                if hasattr(model, '_current_temperature'):
                    model._current_temperature = temperature
                if hasattr(model, '_current_epoch'):
                    model._current_epoch = epoch

                # ── Telemetry: epoch start ─────────────────────────────────
                lib.ce_collector_start_epoch(self._collector, epoch)

                # ── Reseed generator for data shuffling each epoch ───────────
                # The generator is shared with the dataloader's prefetch
                # worker, which reads it in ce_alu_gen_batch() — mutating
                # rng_state/n_ops/ops directly was a data race.  Route every
                # reconfiguration through ce_dataloader_reseed(), which joins
                # the worker, applies the new seed/op filter, discards
                # batches generated under the old settings and restarts
                # prefetch, so the batch fetched below uses the new settings.
                ops_ptr = None
                n_ops = 0
                if cfg.use_curriculum and hasattr(model, '_curriculum_ops'):
                    allowed = sorted(model._curriculum_ops)
                    if len(allowed) > 0:
                        n_ops = len(allowed)
                        ops_ptr = (c_int * n_ops)(*allowed)
                if self._dataloader is not None:
                    check(lib.ce_dataloader_reseed(
                        self._dataloader, cfg.seed + epoch, ops_ptr, n_ops))

                # ── Data generation ────────────────────────────────────────
                lib.ce_collector_start_data_gen(self._collector)
                batch_ptr = lib.ce_dataloader_next(self._dataloader)
                if not batch_ptr:
                    raise CEngineError("ce_dataloader_next returned NULL")
                lib.ce_collector_end_data_gen(self._collector)

                # ── Prepare tensors ────────────────────────────────────────
                max_in = self._in_dim if hasattr(self, '_in_dim') and self._in_dim else 0
                max_tgt = self._out_dim if hasattr(self, '_out_dim') and self._out_dim else 0
                # Detect recurrent model: in_dim=6 means per-step input
                is_recurrent = (self._in_dim is not None and self._in_dim == 6)
                input_tensor, target_tensor, n_bits, mask_tensor = _batch_to_tensors(
                    batch_ptr, max_input_dim=max_in, max_target_dim=max_tgt,
                    recurrent=is_recurrent,
                )

                batch_count = batch_ptr.contents.count
                in_ndim = input_tensor.contents.ndim

                if in_ndim == 3:
                    # Recurrent model: [batch, seq_len, features]
                    seq_len = input_tensor.contents.shape[1]
                    in_feat = input_tensor.contents.shape[2]
                    out_feat = self._out_dim if self._out_dim else 2

                    # Output: [batch, seq_len, out_feat]
                    out_shape = (c_int64 * 3)(batch_count, seq_len, out_feat)
                    output_tensor = lib.ce_tensor_create(out_shape, 3, CE_FLOAT32)

                    # Pad/truncate target to match output
                    tgt_feat = target_tensor.contents.shape[2]
                    if tgt_feat != out_feat:
                        padded_shape = (c_int64 * 3)(batch_count, seq_len, out_feat)
                        padded_target = lib.ce_tensor_zeros(padded_shape, 3)
                        copy_dim = min(int(tgt_feat), out_feat)
                        src_data = lib.ce_tensor_data_ptr(target_tensor)
                        dst_data = lib.ce_tensor_data_ptr(padded_target)
                        for _bi in range(batch_count):
                            for _sl in range(seq_len):
                                src_off = (_bi * seq_len + _sl) * int(tgt_feat)
                                dst_off = (_bi * seq_len + _sl) * out_feat
                                ctypes.memmove(
                                    ctypes.addressof(dst_data.contents) + dst_off * ctypes.sizeof(ctypes.c_float),
                                    ctypes.addressof(src_data.contents) + src_off * ctypes.sizeof(ctypes.c_float),
                                    copy_dim * ctypes.sizeof(ctypes.c_float),
                                )
                        lib.ce_tensor_release(target_tensor)
                        target_tensor = padded_target

                    # Grad output: [batch, seq_len, out_feat]
                    grad_tensor = lib.ce_tensor_zeros(out_shape, 3)
                else:
                    # Flat model: [batch, features]
                    out_dim = self._out_dim if self._out_dim else 2 * n_bits
                    out_shape = (c_int64 * 2)(batch_count, out_dim)
                    output_tensor = lib.ce_tensor_create(out_shape, 2, CE_FLOAT32)

                    # Pad/truncate target
                    tgt_ndim = target_tensor.contents.ndim
                    tgt_dim = target_tensor.contents.shape[tgt_ndim - 1]
                    if tgt_dim != out_dim:
                        padded_shape = (c_int64 * 2)(batch_count, out_dim)
                        padded_target = lib.ce_tensor_zeros(padded_shape, 2)
                        copy_dim = min(int(tgt_dim), out_dim)
                        src_data = lib.ce_tensor_data_ptr(target_tensor)
                        dst_data = lib.ce_tensor_data_ptr(padded_target)
                        for _bi in range(batch_count):
                            ctypes.memmove(
                                ctypes.addressof(dst_data.contents) + _bi * out_dim * ctypes.sizeof(ctypes.c_float),
                                ctypes.addressof(src_data.contents) + _bi * int(tgt_dim) * ctypes.sizeof(ctypes.c_float),
                                copy_dim * ctypes.sizeof(ctypes.c_float),
                            )
                        lib.ce_tensor_release(target_tensor)
                        target_tensor = padded_target

                    grad_tensor = lib.ce_tensor_clone(output_tensor)

                # ── Forward ────────────────────────────────────────────────
                lib.ce_collector_start_forward(self._collector)
                model.forward(input_tensor, output_tensor)
                lib.ce_collector_end_forward(self._collector)

                # ── Per-Operation Counts (Diagnostics) ────────────────────
                if in_ndim == 3:
                    op_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                    for _bi in range(batch_count):
                        _ob0 = int(input_tensor.contents.data[(_bi * int(seq_len) + 0) * 6 + 3])
                        _ob1 = int(input_tensor.contents.data[(_bi * int(seq_len) + 0) * 6 + 4])
                        _ob2 = int(input_tensor.contents.data[(_bi * int(seq_len) + 0) * 6 + 5])
                        op_counts[_ob0 | (_ob1 << 1) | (_ob2 << 2)] += 1
                    self._last_op_counts = op_counts

                # ── Loss ───────────────────────────────────────────────────
                if cfg.loss_type == "bce":
                    loss_val = lib.ce_bce_with_logits_loss(
                        output_tensor, target_tensor, mask_tensor,
                    )
                else:
                    loss_val = lib.ce_mse_loss(
                        output_tensor, target_tensor, mask_tensor,
                    )

                # ── Router auxiliary loss (balance + z-loss) ──────────────
                # Adds regularisation pressure that keeps logits bounded
                # and encourages uniform expert usage. The gradient of this
                # loss is computed analytically inside model.backward().
                router_loss = 0.0
                if hasattr(model, 'router_loss'):
                    router_loss = model.router_loss(
                        cfg.lambda_balance, cfg.lambda_z)

                total_loss = loss_val + router_loss
                self._current_loss = total_loss
                self._last_router_loss = router_loss

                # NaN check on loss
                if loss_val != loss_val:  # NaN check
                    print(f"  NaN loss at epoch {epoch} — stopping training")
                    # Close the epoch's telemetry pair (start_epoch ran at
                    # the top of the loop): the break below exits the whole
                    # run, so without this the started epoch would silently
                    # vanish from every consumer.
                    lib.ce_collector_end_epoch(self._collector)
                    lib.ce_tensor_release(input_tensor)
                    lib.ce_tensor_release(target_tensor)
                    lib.ce_tensor_release(mask_tensor)
                    lib.ce_tensor_release(output_tensor)
                    lib.ce_tensor_release(grad_tensor)
                    lib.ce_batch_destroy(batch_ptr)
                    break

                # ── Backward ───────────────────────────────────────────────
                lib.ce_collector_start_backward(self._collector)

                # Allocate gradient tensor for loss
                grad_tensor = lib.ce_tensor_clone(output_tensor)
                if cfg.loss_type == "bce":
                    # Verify shapes match before backward
                    out_s = output_tensor.contents
                    tgt_s = target_tensor.contents
                    if out_s.shape[0] != tgt_s.shape[0] or out_s.shape[1] != tgt_s.shape[1]:
                        raise CEngineError(
                            f"Shape mismatch in BCE backward: "
                            f"output=[{out_s.shape[0]},{out_s.shape[1]}] "
                            f"target=[{tgt_s.shape[0]},{tgt_s.shape[1]}]"
                        )
                    check(lib.ce_bce_with_logits_backward(
                        output_tensor, target_tensor, mask_tensor, grad_tensor,
                    ))
                else:
                    check(lib.ce_mse_backward(
                        output_tensor, target_tensor, mask_tensor, grad_tensor,
                    ))

                # Allocate grad_input with correct shape
                if in_ndim == 3:
                    grad_in_shape = (c_int64 * 3)(batch_count, input_tensor.contents.shape[1], self._in_dim)
                    grad_input_tensor = lib.ce_tensor_zeros(grad_in_shape, 3)
                else:
                    grad_in_shape = (c_int64 * 2)(batch_count, self._in_dim)
                    grad_input_tensor = lib.ce_tensor_zeros(grad_in_shape, 2)
                model.backward(grad_tensor, grad_input_tensor)
                lib.ce_tensor_release(grad_input_tensor)

                # Release MoE expert outputs (if model has cleanup method)
                if hasattr(model, 'cleanup_expert_outputs'):
                    model.cleanup_expert_outputs()

                lib.ce_collector_end_backward(self._collector)

                # ── Gradient clipping ──────────────────────────────────────
                if cfg.grad_clip_norm > 0:
                    grad_norm = lib.ce_compute_grad_norm(
                        self._grads_arr, self._n_grads,
                    )
                    # NaN check on gradients
                    if grad_norm != grad_norm:  # NaN check
                        print(f"  WARNING: NaN gradient at epoch {epoch}, skipping step")
                        # Close the epoch pair before skipping: start_epoch
                        # ran at the top of the loop, and the continue below
                        # bypasses the normal end_epoch.
                        lib.ce_collector_end_epoch(self._collector)
                        lib.ce_tensor_release(input_tensor)
                        lib.ce_tensor_release(target_tensor)
                        lib.ce_tensor_release(mask_tensor)
                        lib.ce_tensor_release(output_tensor)
                        lib.ce_tensor_release(grad_tensor)
                        lib.ce_batch_destroy(batch_ptr)
                        continue
                    check(lib.ce_clip_grad_norm(
                        self._grads_arr, self._n_grads, c_float(cfg.grad_clip_norm),
                    ))
                else:
                    grad_norm = 0.0

                # ── Optimizer step ─────────────────────────────────────────
                lib.ce_collector_start_optimizer(self._collector)
                if self._opt_type == "adam":
                    check(lib.ce_adam_step(self._optimizer))
                else:
                    check(lib.ce_sgd_step(self._optimizer))

                # Sync Python weights from C after optimizer update
                if hasattr(model, 'sync_all_from_c'):
                    model.sync_all_from_c()

                # Scheduler step
                new_lr = lib.ce_scheduler_step(self._scheduler)
                lib.ce_collector_end_optimizer(self._collector)

                # ── Curriculum update ──────────────────────────────────────
                if self._curriculum is not None:
                    stage_changed = lib.ce_curriculum_update(
                        self._curriculum, c_float(loss_val),
                    )
                    if stage_changed:
                        # Recreate generator and dataloader for new bit range
                        if self._dataloader:
                            lib.ce_dataloader_destroy(self._dataloader)
                        # The dataloader does NOT own the generator — free
                        # the old one before replacing it (leaked per stage
                        # change before this).
                        if self._generator:
                            lib.ce_alu_gen_destroy(self._generator)
                        self._generator = lib.ce_curriculum_make_generator(
                            self._curriculum, c_int64(cfg.seed + epoch),
                        )
                        self._dataloader = lib.ce_dataloader_create(
                            self._generator,
                            c_int(cfg.batch_size),
                            c_int(cfg.samples_per_epoch),
                            c_int64(cfg.seed + epoch),
                        )

                # ── Telemetry: record metrics ──────────────────────────────
                lib.ce_collector_record_loss(
                    self._collector,
                    c_float(total_loss), c_float(loss_val),  # total, result
                    c_float(0.0), c_float(router_loss), 0.0,  # carry, balance, z
                )
                lib.ce_collector_record_lr(
                    self._collector, c_float(new_lr), c_float(temperature),
                )
                lib.ce_collector_record_grad_norm(self._collector, c_float(grad_norm))

                # ── Debug: log gradient norms per parameter ──────────────
                if cfg.enable_debug_logging and epoch % max(cfg.save_every_n_epochs, 1) == 0:
                    grads = model.gradients()
                    total_norm_sq = 0.0
                    for gi, g in enumerate(grads):
                        n = g.contents.numel
                        arr = (ctypes.c_float * n).from_address(
                            ctypes.addressof(g.contents.data.contents)
                        )
                        norm_sq = sum(arr[j] * arr[j] for j in range(n))
                        total_norm_sq += norm_sq
                        norm = norm_sq ** 0.5
                        print(f"  [GradDebug] param[{gi}] norm={norm:.6f}", file=sys.stderr)
                    import math
                    print(f"  [GradDebug] total_norm={math.sqrt(total_norm_sq):.6f}",
                          file=sys.stderr)

                mem_stats = lib.ce_tracker_get_stats()
                lib.ce_collector_record_memory(
                    self._collector, ctypes.pointer(mem_stats),
                )

                if self._curriculum is not None:
                    stage_name = lib.ce_curriculum_stage_name(self._curriculum)
                    stage_idx = self._curriculum.contents.current_stage
                    lib.ce_collector_record_curriculum(self._collector, stage_idx)

                lib.ce_collector_end_epoch(self._collector)

                # ── Free temporary tensors ─────────────────────────────────
                lib.ce_tensor_release(input_tensor)
                lib.ce_tensor_release(target_tensor)
                lib.ce_tensor_release(mask_tensor)
                lib.ce_tensor_release(output_tensor)
                lib.ce_tensor_release(grad_tensor)
                lib.ce_batch_destroy(batch_ptr)

                # ── Best loss tracking ─────────────────────────────────────
                if total_loss < self._best_loss:
                    self._best_loss = total_loss

                # ── Checkpoint ─────────────────────────────────────────────
                if (
                    cfg.save_every_n_epochs > 0
                    and (epoch + 1) % cfg.save_every_n_epochs == 0
                ):
                    self._save_checkpoint(epoch, total_loss)

                # ── User callback ──────────────────────────────────────────
                if self._on_epoch_end is not None:
                    self._on_epoch_end(epoch, total_loss, self)

                # ── Training state channel (for TUI) ────────────────────
                if self._training_state is not None:
                    try:
                        mem_stats_cur = lib.ce_tracker_get_stats()
                        self._training_state.update(
                            epoch=epoch + 1,
                            total_epochs=cfg.total_epochs,
                            loss=total_loss,
                            best_loss=self._best_loss,
                            learning_rate=new_lr,
                            temperature=temperature,
                            grad_norm=grad_norm,
                            memory_mb=mem_stats_cur.current_bytes / (1024 * 1024),
                            status="training",
                        )
                    except Exception:
                        pass

        except Exception as exc:
            self._running = False
            raise
        finally:
            self._running = False
            self._teardown()

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save checkpoint: binary weights + JSON metadata."""
        cfg = self._config
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # Save binary weights
        self.save_model(cfg.checkpoint_dir)

        # Save JSON metadata alongside
        import json
        path = os.path.join(cfg.checkpoint_dir, f"checkpoint_epoch_{epoch:06d}.json")
        data = {
            "epoch": epoch,
            "loss": loss,
            "best_loss": self._best_loss,
            "config": cfg.to_c_params(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_model(self, path: str) -> None:
        """Save model weights to a directory.

        Creates ``path/checkpoint_latest.bin`` (binary) and ``meta.json``.
        Works with any Module — uses parameter tensors directly.
        """
        import json
        os.makedirs(path, exist_ok=True)

        params = self._model.parameters()
        n_params = len(params)

        # Try C checkpoint first (works if model has a C handle)
        c_handle = getattr(self._model, '_handle', None)
        if c_handle is not None:
            status = lib.ce_checkpoint_save(
                path.encode("utf-8"),
                self._current_epoch,
                c_float(self._current_loss),
                c_handle,
                None,
            )
            if status == 0:
                # Also save meta
                meta = {
                    "epoch": self._current_epoch,
                    "loss": float(self._current_loss),
                    "best_loss": float(self._best_loss),
                    "n_params": n_params,
                    "method": "c_checkpoint",
                }
                with open(os.path.join(path, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                return

        # Fallback: save parameters as raw binary blobs
        import struct
        bin_path = os.path.join(path, "checkpoint_latest.bin")
        with open(bin_path, "wb") as f:
            # Header: magic + version + n_params
            f.write(b"CEPY")
            f.write(struct.pack("<I", 1))  # version
            f.write(struct.pack("<I", n_params))
            for p in params:
                ndim = p.contents.ndim
                f.write(struct.pack("<I", ndim))
                for d in range(ndim):
                    f.write(struct.pack("<I", p.contents.shape[d]))
                numel = p.contents.numel
                f.write(struct.pack("<Q", numel))
                # Write raw float data
                data_addr = ctypes.addressof(p.contents.data.contents)
                f.write(ctypes.string_at(data_addr, numel * 4))

        meta = {
            "epoch": self._current_epoch,
            "loss": float(self._current_loss),
            "best_loss": float(self._best_loss),
            "n_params": n_params,
            "method": "python_raw",
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load_model(self, path: str) -> None:
        """Load model weights from a checkpoint directory.

        Supports both C checkpoint format and Python raw format.
        """
        import json

        # Try C checkpoint first
        c_handle = getattr(self._model, '_handle', None)
        if c_handle is not None:
            status = lib.ce_checkpoint_load(
                path.encode("utf-8"),
                c_handle,
            )
            if status == 0:
                # Load meta if available
                meta_path = os.path.join(path, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    self._current_epoch = meta.get("epoch", 0)
                    self._current_loss = meta.get("loss", 0.0)
                    self._best_loss = meta.get("best_loss", 0.0)
                return

        # Fallback: load Python raw format — two-phase: parse and validate
        # the WHOLE file against the model BEFORE touching any tensor.
        # The old code trusted the file's own numel/shape: a different
        # layout was silently loaded (transposed weights), an oversized
        # numel memmove'd past the model tensor (heap OOB write), and a
        # truncated file produced a SHORT buffer memmove read past —
        # mixed weights with no rollback either way.
        import struct
        bin_path = os.path.join(path, "checkpoint_latest.bin")
        if not os.path.exists(bin_path):
            print(f"  WARN: No checkpoint found at {path}")
            return

        with open(bin_path, "rb") as f:
            blob = f.read()

        if len(blob) < 12 or blob[:4] != b"CEPY":
            raise CEngineError(
                f"Unrecognised checkpoint format in {bin_path!r}: "
                f"{blob[:4] if len(blob) >= 4 else blob!r}"
            )
        n_params = struct.unpack_from("<I", blob, 8)[0]
        params = self._model.parameters()
        if n_params != len(params):
            raise CEngineError(
                f"Param count mismatch: checkpoint={n_params}, "
                f"model={len(params)}"
            )

        pos = 12
        staged: list[tuple[Any, bytes]] = []
        for idx, p in enumerate(params):
            if pos + 4 > len(blob):
                raise CEngineError(f"Truncated checkpoint before param {idx}")
            ndim = struct.unpack_from("<I", blob, pos)[0]
            pos += 4
            if pos + 4 * ndim + 8 > len(blob):
                raise CEngineError(f"Truncated checkpoint header of param {idx}")
            shape = list(struct.unpack_from(f"<{ndim}I", blob, pos))
            pos += 4 * ndim
            numel = struct.unpack_from("<Q", blob, pos)[0]
            pos += 8

            model_shape = [p.contents.shape[d] for d in range(p.contents.ndim)]
            if ndim != p.contents.ndim or shape != model_shape \
                    or numel != p.contents.numel:
                raise CEngineError(
                    f"Checkpoint shape mismatch for param {idx}: "
                    f"file ndim={ndim} shape={shape} numel={numel}, "
                    f"model ndim={p.contents.ndim} shape={model_shape} "
                    f"numel={p.contents.numel}"
                )
            if pos + numel * 4 > len(blob):
                raise CEngineError(f"Truncated data of param {idx}")
            staged.append((p, blob[pos:pos + numel * 4]))
            pos += numel * 4

        # Commit — only after every parameter validated (all-or-nothing).
        for p, raw in staged:
            ctypes.memmove(ctypes.addressof(p.contents.data.contents),
                           raw, len(raw))

        # Load meta
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self._current_epoch = meta.get("epoch", 0)
            self._current_loss = meta.get("loss", 0.0)
            self._best_loss = meta.get("best_loss", 0.0)

    # ── PyTorch checkpoint ────────────────────────────────────────────────

    # Parameter name mapping for RecurrentExpert (matches C registration order)
    _RECURRENT_PARAM_NAMES = [
        "w1", "b1", "w2", "b2", "w_out", "b_out", "w_h", "b_h",
    ]

    def save_model_pytorch(self, path: str) -> None:
        """Save model weights in PyTorch-compatible .pt format.

        Creates ``path`` as a .pt file with state_dict + metadata.
        """
        import torch
        import json

        params = self._model.parameters()
        state_dict = {}

        # Try to determine parameter names from model structure
        try:
            names = self._infer_param_names()
        except Exception:
            names = [f"param_{i}" for i in range(len(params))]

        for name, p in zip(names, params):
            ndim = p.contents.ndim
            shape = tuple(p.contents.shape[d] for d in range(ndim))
            numel = p.contents.numel
            data_addr = ctypes.addressof(p.contents.data.contents)
            raw = ctypes.string_at(data_addr, numel * 4)
            # Create writable tensor from raw bytes
            buf = (ctypes.c_float * numel).from_address(data_addr)
            tensor = torch.tensor(buf, dtype=torch.float32).reshape(shape)
            state_dict[name] = tensor

        metadata = {
            "epoch": self._current_epoch,
            "loss": float(self._current_loss),
            "best_loss": float(self._best_loss),
            "model_type": type(self._model).__name__,
        }
        torch.save({"state_dict": state_dict, "metadata": metadata}, path)

    def load_model_pytorch(self, path: str) -> None:
        """Load model weights from a PyTorch .pt checkpoint.

        Two-phase like the raw/CEPY loader: every parameter (name count,
        key presence, size) is validated BEFORE any tensor is touched —
        a bad checkpoint raises CEngineError with the model left
        untouched instead of silently ending up with a MIX of loaded
        and initial weights (the old code skipped bad entries with a
        printed WARN and kept going).

        Args:
            path: path to .pt file
        """
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        params = self._model.parameters()
        try:
            names = self._infer_param_names()
        except Exception:
            names = list(state_dict.keys())
        if len(names) != len(params):
            raise CEngineError(
                f"Name/model mismatch: {len(names)} names for "
                f"{len(params)} parameters"
            )

        staged = []
        problems = []
        for name, p in zip(names, params):
            if name not in state_dict:
                problems.append(f"key {name!r} missing")
                continue
            t = state_dict[name].float().contiguous()
            numel = p.contents.numel
            if t.numel() != numel:
                problems.append(
                    f"size mismatch for {name}: "
                    f"checkpoint={t.numel()}, model={numel}"
                )
                continue
            staged.append((p, t))

        if problems:
            raise CEngineError(
                "PyTorch checkpoint rejected (model left untouched): "
                + "; ".join(problems)
            )

        # Commit — only after every parameter validated (all-or-nothing).
        for p, t in staged:
            ctypes.memmove(ctypes.addressof(p.contents.data.contents),
                           t.data_ptr(), p.contents.numel * 4)

        # Restore metadata
        meta = checkpoint.get("metadata", {})
        self._current_epoch = meta.get("epoch", 0)
        self._current_loss = meta.get("loss", 0.0)
        self._best_loss = meta.get("best_loss", 0.0)

    def _infer_param_names(self) -> list[str]:
        """Infer parameter names from model structure."""
        params = self._model.parameters()
        n = len(params)

        # RecurrentExpert has exactly 8 params
        if n == 8:
            return list(self._RECURRENT_PARAM_NAMES)

        # MoE with E experts: E * 8 params
        if hasattr(self._model, 'experts') and n == len(self._model.experts) * 8:
            names = []
            for i, _ in enumerate(self._model.experts):
                for pname in self._RECURRENT_PARAM_NAMES:
                    names.append(f"expert.{i}.{pname}")
            return names

        # Fallback: generic names
        return [f"param_{i}" for i in range(n)]

    # ── Repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "running" if self._running else "idle"
        return (
            f"CEngineTrainer(status={status}, epoch={self._current_epoch}, "
            f"loss={self._current_loss:.6f}, best={self._best_loss:.6f})"
        )

    def __del__(self) -> None:
        if self._running:
            self._stop_requested = True
            self.join(timeout=5.0)
        # _teardown is called in _run_loop's finally block
