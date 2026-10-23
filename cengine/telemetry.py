"""
Telemetry monitoring: ring-buffer polling, stats, leak detection, live plotting.

The C engine pushes ``ce_metric_t`` records into a lock-free SPSC ring buffer.
The ``TelemetryMonitor`` drains that buffer from Python, builds a history of
every metric type, computes rolling statistics, and can export or plot the data.

Usage::

    monitor = trainer.get_telemetry()
    while trainer.is_running:
        metrics = monitor.poll()
        if monitor.detect_memory_leak():
            print("WARNING: memory leak detected")
        time.sleep(0.5)

    monitor.export_csv("telemetry.csv")
"""

from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .bindings import (
    CE_METRIC_COUNT,
    CE_METRIC_CPU_UTIL,
    CE_METRIC_CURRICULUM_STAGE,
    CE_METRIC_DATA_GEN_TIME_MS,
    CE_METRIC_EPOCH_TIME_MS,
    CE_METRIC_EXPERT_USAGE,
    CE_METRIC_FORWARD_TIME_MS,
    CE_METRIC_GRAD_NORM,
    CE_METRIC_HW_BRANCH_MISS_RATE,
    CE_METRIC_HW_IPC,
    CE_METRIC_HW_L2_MISS_RATE,
    CE_METRIC_HW_LLC_MISS_RATE,
    CE_METRIC_INF_COUNT,
    CE_METRIC_LOSS_BALANCE,
    CE_METRIC_LOSS_CARRY,
    CE_METRIC_LOSS_RESULT,
    CE_METRIC_LOSS_TOTAL,
    CE_METRIC_LOSS_Z,
    CE_METRIC_LR,
    CE_METRIC_MEM_CURRENT,
    CE_METRIC_MEM_GROWTH_RATE,
    CE_METRIC_MEM_PEAK,
    CE_METRIC_NAN_COUNT,
    CE_METRIC_OPTIMIZER_TIME_MS,
    CE_METRIC_BACKWARD_TIME_MS,
    CE_METRIC_TEMPERATURE,
    CE_METRIC_THROUGHPUT,
    CE_METRIC_THREADPOOL_QUEUE_DEPTH,
    CE_METRIC_ACTIVE_WORKERS,
    CE_METRIC_FLAG_VALID,
    CeMemoryStats,
    CeMetric,
    CeReporter,
    CeRingBuffer,
    CeTelemetrySnapshot,
    CEngineError,
    check,
    lib,
)
from ctypes import POINTER, c_float, byref

__all__ = ["TelemetryMonitor", "MetricPoint", "MemoryStats"]


# ---------------------------------------------------------------------------
# Friendly names for metric types
# ---------------------------------------------------------------------------

METRIC_NAMES: dict[int, str] = {
    CE_METRIC_LOSS_TOTAL: "loss_total",
    CE_METRIC_LOSS_RESULT: "loss_result",
    CE_METRIC_LOSS_CARRY: "loss_carry",
    CE_METRIC_LOSS_BALANCE: "loss_balance",
    CE_METRIC_LOSS_Z: "loss_z",
    CE_METRIC_LR: "lr",
    CE_METRIC_TEMPERATURE: "temperature",
    CE_METRIC_MEM_CURRENT: "mem_current",
    CE_METRIC_MEM_PEAK: "mem_peak",
    CE_METRIC_MEM_GROWTH_RATE: "mem_growth_rate",
    CE_METRIC_GRAD_NORM: "grad_norm",
    CE_METRIC_EXPERT_USAGE: "expert_usage",
    CE_METRIC_THROUGHPUT: "throughput",
    CE_METRIC_EPOCH_TIME_MS: "epoch_time_ms",
    CE_METRIC_FORWARD_TIME_MS: "forward_time_ms",
    CE_METRIC_BACKWARD_TIME_MS: "backward_time_ms",
    CE_METRIC_OPTIMIZER_TIME_MS: "optimizer_time_ms",
    CE_METRIC_DATA_GEN_TIME_MS: "data_gen_time_ms",
    CE_METRIC_CPU_UTIL: "cpu_util",
    CE_METRIC_HW_IPC: "hw_ipc",
    CE_METRIC_HW_L2_MISS_RATE: "hw_l2_miss_rate",
    CE_METRIC_HW_LLC_MISS_RATE: "hw_llc_miss_rate",
    CE_METRIC_HW_BRANCH_MISS_RATE: "hw_branch_miss_rate",
    CE_METRIC_ACTIVE_WORKERS: "active_workers",
    CE_METRIC_THREADPOOL_QUEUE_DEPTH: "threadpool_queue_depth",
    CE_METRIC_NAN_COUNT: "nan_count",
    CE_METRIC_INF_COUNT: "inf_count",
    CE_METRIC_CURRICULUM_STAGE: "curriculum_stage",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricPoint:
    """A single decoded metric record."""

    metric_type: int
    name: str
    epoch: int
    timestamp_ns: int
    value: float | list[float] | int
    valid: bool = True
    alert: bool = False


@dataclass
class MemoryStats:
    """Python-friendly mirror of ``ce_memory_stats_t``."""

    current_bytes: int = 0
    peak_bytes: int = 0
    total_allocs: int = 0
    total_frees: int = 0
    active_allocs: int = 0
    alloc_bytes_rate: int = 0
    free_bytes_rate: int = 0
    growth_rate: float = 0.0
    fragmentation: float = 0.0
    tensor_count: int = 0
    tensor_total_bytes: int = 0

    @classmethod
    def from_c(cls, c_stats: CeMemoryStats) -> MemoryStats:
        return cls(
            current_bytes=c_stats.current_bytes,
            peak_bytes=c_stats.peak_bytes,
            total_allocs=c_stats.total_allocs,
            total_frees=c_stats.total_frees,
            active_allocs=c_stats.active_allocs,
            alloc_bytes_rate=c_stats.alloc_bytes_rate,
            free_bytes_rate=c_stats.free_bytes_rate,
            growth_rate=c_stats.growth_rate,
            fragmentation=c_stats.fragmentation,
            tensor_count=c_stats.tensor_count,
            tensor_total_bytes=c_stats.tensor_total_bytes,
        )

    @property
    def current_mb(self) -> float:
        return self.current_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"MemoryStats(current={self.current_mb:.1f}MB, peak={self.peak_mb:.1f}MB, "
            f"tensors={self.tensor_count}, active_allocs={self.active_allocs})"
        )


# ---------------------------------------------------------------------------
# Telemetry Monitor
# ---------------------------------------------------------------------------

class TelemetryMonitor:
    """Polls the C ring buffer, builds history, detects leaks, exports data.

    Args:
        ring_buffer_ptr: ``POINTER(CeRingBuffer)`` from the C engine.
        reporter_ptr: ``POINTER(CeReporter)`` for snapshot-based polling.
    """

    def __init__(
        self,
        ring_buffer_ptr: POINTER(CeRingBuffer),
        reporter_ptr: POINTER(CeReporter) | None = None,
    ) -> None:
        self._ring = ring_buffer_ptr
        self._reporter = reporter_ptr
        self._history: dict[str, list[MetricPoint]] = defaultdict(list)
        self._latest: dict[str, MetricPoint] = {}
        self._callbacks: list[Callable[[MetricPoint], None]] = []
        self._lock = threading.Lock()

    # ── Ring buffer polling ───────────────────────────────────────────────

    def poll(self, max_items: int = 4096) -> list[MetricPoint]:
        """Non-blocking drain of up to *max_items* from the ring buffer.

        Returns a list of decoded ``MetricPoint`` objects.  Points are also
        appended to the internal history.
        """
        buf = (CeMetric * max_items)()
        n = lib.ce_ring_batch_pop(self._ring, buf, max_items)
        if n <= 0:
            return []

        points: list[MetricPoint] = []
        with self._lock:
            for i in range(n):
                m = buf[i]
                pt = self._decode_metric(m)
                points.append(pt)
                name = pt.name
                self._history[name].append(pt)
                self._latest[name] = pt
                for cb in self._callbacks:
                    cb(pt)
        return points

    def poll_snapshot(self) -> dict[str, Any]:
        """Use the reporter to get a single aggregated snapshot dict."""
        if self._reporter is None:
            raise CEngineError("No reporter attached — use poll() instead")
        lib.ce_reporter_poll(self._reporter)
        snap = lib.ce_reporter_get_snapshot(self._reporter)
        return self._snapshot_to_dict(snap)

    # ── Latest / history access ───────────────────────────────────────────

    def get_latest(self) -> dict[str, MetricPoint]:
        """Return the most recent ``MetricPoint`` for each metric name."""
        with self._lock:
            return dict(self._latest)

    def get_latest_values(self) -> dict[str, float | int | list[float]]:
        """Convenience: name → raw value for every tracked metric."""
        with self._lock:
            return {name: pt.value for name, pt in self._latest.items()}

    def get_history(self) -> dict[str, list[MetricPoint]]:
        """Return the full history keyed by metric name."""
        with self._lock:
            return {k: list(v) for k, v in self._history.items()}

    def get_history_values(self, name: str) -> list[float]:
        """Return a flat list of float values for a single metric."""
        with self._lock:
            return [
                pt.value if isinstance(pt.value, (int, float)) else pt.value[0]
                for pt in self._history.get(name, [])
            ]

    # ── Callbacks ─────────────────────────────────────────────────────────

    def on_metric(self, callback: Callable[[MetricPoint], None]) -> None:
        """Register a callback invoked for every decoded metric."""
        self._callbacks.append(callback)

    # ── Memory leak detection ─────────────────────────────────────────────

    def detect_memory_leak(
        self,
        window: int = 100,
        threshold: float = 0.01,
    ) -> bool:
        """Analyse the memory growth-rate trend over the last *window* samples.

        Returns ``True`` if the average growth rate exceeds *threshold*
        (fraction of current usage per sample).  A simple linear-regression
        slope is used.
        """
        values = self.get_history_values("mem_current")
        if len(values) < window:
            return False

        recent = values[-window:]
        n = len(recent)
        if n < 2:
            return False

        # Simple linear regression: slope of mem_current vs sample index
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        ss_xy = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        ss_xx = sum((i - x_mean) ** 2 for i in range(n))
        if ss_xx == 0:
            return False
        slope = ss_xy / ss_xx

        # Normalise slope by mean to get fractional growth
        if y_mean <= 0:
            return False
        fractional_growth = slope / y_mean
        return fractional_growth > threshold

    def detect_anomalies(self) -> dict[str, int]:
        """Return counts of NaN and Inf values seen."""
        nan_pts = self._history.get("nan_count", [])
        inf_pts = self._history.get("inf_count", [])
        return {
            "nan_count": int(nan_pts[-1].value) if nan_pts else 0,
            "inf_count": int(inf_pts[-1].value) if inf_pts else 0,
        }

    # ── Statistics ────────────────────────────────────────────────────────

    def compute_stats(self, name: str) -> dict[str, float]:
        """Compute min / max / mean / stddev for a named metric."""
        values = self.get_history_values(name)
        if not values:
            return {}
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
        return {
            "count": n,
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "stddev": math.sqrt(var),
            "latest": values[-1],
        }

    # ── Export ─────────────────────────────────────────────────────────────

    def export_json(self, path: str) -> None:
        """Export all history to a JSON file."""
        data: dict[str, list[dict]] = {}
        with self._lock:
            for name, points in self._history.items():
                data[name] = [
                    {
                        "epoch": pt.epoch,
                        "timestamp_ns": pt.timestamp_ns,
                        "value": pt.value if isinstance(pt.value, (int, float)) else list(pt.value),
                    }
                    for pt in points
                ]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, path: str) -> None:
        """Export all history to a CSV file (one column per metric)."""
        with self._lock:
            all_names = sorted(self._history.keys())
            if not all_names:
                return
            # Determine max length
            max_len = max(len(v) for v in self._history.values())

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index"] + all_names)
            with self._lock:
                for i in range(max_len):
                    row = [i]
                    for name in all_names:
                        pts = self._history[name]
                        if i < len(pts):
                            val = pts[i].value
                            row.append(val if isinstance(val, (int, float)) else str(val))
                        else:
                            row.append("")
                    writer.writerow(row)

    # ── Live plotting (optional matplotlib) ───────────────────────────────

    def plot_live(self, interval_ms: int = 500, max_points: int = 500) -> None:
        """Launch a real-time matplotlib dashboard (blocking).

        Requires ``matplotlib``.  Closes on window close or Ctrl-C.
        """
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_live().  "
                "Install it with: pip install matplotlib"
            )

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("CEngine Telemetry", fontsize=14)
        fig.tight_layout(pad=3.0)

        ax_loss, ax_lr, ax_mem, ax_time = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        def _update(_frame: int) -> None:
            self.poll(max_items=4096)

            for ax in axes.flat:
                ax.clear()

            # Loss
            for name in ("loss_total", "loss_result", "loss_carry"):
                vals = self.get_history_values(name)[-max_points:]
                if vals:
                    ax_loss.plot(vals, label=name, linewidth=0.8)
            ax_loss.set_title("Loss")
            ax_loss.legend(fontsize=8)
            ax_loss.set_yscale("log")

            # LR + Temperature
            lr_vals = self.get_history_values("lr")[-max_points:]
            if lr_vals:
                ax_lr.plot(lr_vals, label="lr", color="tab:blue")
                ax_lr.set_ylabel("Learning Rate", color="tab:blue")
            temp_vals = self.get_history_values("temperature")[-max_points:]
            if temp_vals:
                ax2 = ax_lr.twinx()
                ax2.plot(temp_vals, label="temp", color="tab:orange", linewidth=0.8)
                ax2.set_ylabel("Temperature", color="tab:orange")
            ax_lr.set_title("LR & Temperature")

            # Memory
            mem_vals = self.get_history_values("mem_current")[-max_points:]
            if mem_vals:
                ax_mem.plot(mem_vals, color="tab:green")
            ax_mem.set_title("Memory (MB)")
            ax_mem.set_ylabel("MB")

            # Timing
            for name, color in [
                ("epoch_time_ms", "tab:red"),
                ("forward_time_ms", "tab:blue"),
                ("backward_time_ms", "tab:orange"),
            ]:
                vals = self.get_history_values(name)[-max_points:]
                if vals:
                    ax_time.plot(vals, label=name, linewidth=0.8, color=color)
            ax_time.set_title("Timing (ms)")
            ax_time.legend(fontsize=8)

        ani = FuncAnimation(fig, _update, interval=interval_ms, cache_frame_data=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            plt.close(fig)

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _decode_metric(m: CeMetric) -> MetricPoint:
        """Decode a raw ``ce_metric_t`` into a ``MetricPoint``."""
        mt = m.metric_type
        name = METRIC_NAMES.get(mt, f"metric_{mt}")
        valid = bool(m.flags & CE_METRIC_FLAG_VALID)
        alert = bool(m.flags & 0x02)  # CE_METRIC_FLAG_ALERT

        # Expert usage is an array of 4 floats
        if mt == CE_METRIC_EXPERT_USAGE:
            value: Any = list(m.value.f32_arr)
        # Integer metrics
        elif mt in (CE_METRIC_NAN_COUNT, CE_METRIC_INF_COUNT, CE_METRIC_CURRICULUM_STAGE,
                     CE_METRIC_ACTIVE_WORKERS, CE_METRIC_THREADPOOL_QUEUE_DEPTH):
            value = m.value.i64
        else:
            value = m.value.f32

        return MetricPoint(
            metric_type=mt,
            name=name,
            epoch=m.epoch,
            timestamp_ns=m.timestamp_ns,
            value=value,
            valid=valid,
            alert=alert,
        )

    @staticmethod
    def _snapshot_to_dict(snap: CeTelemetrySnapshot) -> dict[str, Any]:
        """Convert a ``ce_telemetry_snapshot_t`` to a plain dict."""
        return {
            "loss_total": snap.loss_total,
            "loss_result": snap.loss_result,
            "loss_carry": snap.loss_carry,
            "loss_balance": snap.loss_balance,
            "loss_z": snap.loss_z,
            "lr": snap.lr,
            "temperature": snap.temperature,
            "mem_current_mb": snap.mem_current_mb,
            "mem_peak_mb": snap.mem_peak_mb,
            "mem_growth_rate": snap.mem_growth_rate,
            "grad_norm": snap.grad_norm,
            "expert_usage": list(snap.expert_usage),
            "throughput": snap.throughput,
            "epoch_time_ms": snap.epoch_time_ms,
            "forward_time_ms": snap.forward_time_ms,
            "backward_time_ms": snap.backward_time_ms,
            "optimizer_time_ms": snap.optimizer_time_ms,
            "data_gen_time_ms": snap.data_gen_time_ms,
            "cpu_util": snap.cpu_util,
            "hw_ipc": snap.hw_ipc,
            "hw_l2_miss_rate": snap.hw_l2_miss_rate,
            "hw_llc_miss_rate": snap.hw_llc_miss_rate,
            "hw_branch_miss_rate": snap.hw_branch_miss_rate,
            "active_workers": snap.active_workers,
            "threadpool_queue_depth": snap.threadpool_queue_depth,
            "nan_count": snap.nan_count,
            "inf_count": snap.inf_count,
            "curriculum_stage": snap.curriculum_stage,
            "current_epoch": snap.current_epoch,
            "last_update_ns": snap.last_update_ns,
        }

    def __repr__(self) -> str:
        n_points = sum(len(v) for v in self._history.values())
        return f"TelemetryMonitor(metrics={len(self._history)}, total_points={n_points})"
