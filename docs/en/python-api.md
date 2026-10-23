# Python API

[English](python-api.md) | [Русский](../ru/python-api.md)

**Docs:** [Getting started](getting-started.md) · [Training](training.md) · Python API · [Architecture](architecture.md)

Everything is importable from the `cengine` package:

```python
from cengine import (
    CEngineTrainer, TrainingConfig,
    Sequential, Linear, ELU, Sigmoid, Tanh, Softmax, GumbelSoftmax,
    RecurrentExpert, TelemetryMonitor,
    lib, check, CEngineError,
)
```

## Modules

`Module` is the base class; containers hold children, leaves wrap a C layer.

| Class | Constructor | Notes |
| --- | --- | --- |
| `Sequential` | `Sequential(*layers, name="")` | chains layers; `add_module(name, module)` |
| `Linear` | `Linear(in_features, out_features, name="")` | Xavier weights, zero bias |
| `ELU` | `ELU(alpha=1.0, name="")` | |
| `Sigmoid` / `Tanh` | `Sigmoid(name="")` | |
| `Softmax` | `Softmax(name="")` | backward = JVP `y·(g − Σ g·y)` |
| `GumbelSoftmax` | `GumbelSoftmax(temperature=1.0, name="")` | |
| `RecurrentExpert` | `RecurrentExpert(hidden_size=64, name=...)` | input `[B, T, 6]`, output `[B, T, 2]` (result, carry) |

Shared `Module` API:

| Member | Meaning |
| --- | --- |
| `forward(input_tensor, output_tensor)` | C forward (ctypes tensors) |
| `backward(grad_output, grad_input)` | C backward |
| `parameters()` / `gradients()` | flat lists of tensor pointers (recursive for containers) |
| `close()` | destroy the C module; safe to call twice; idempotent for children owned by a parent |
| `handle` | underlying `CeModule*` (raises if closed) |

Models are freed automatically on garbage collection, but explicit
`close()` (or relying on the parent container's `close()`) releases the C
side deterministically.

## CEngineTrainer

| Member | Meaning |
| --- | --- |
| `CEngineTrainer(model, config, on_epoch_end=None)` | validate config, prepare C engine |
| `start(blocking=False)` / `stop()` / `join(timeout=None)` | run control |
| `save_model(path)` / `load_model(path)` | checkpoint directory roundtrip |
| `save_model_pytorch(path)` / `load_model_pytorch(path)` | torch state-dict interop (requires `torch`) |
| `get_telemetry()` | `TelemetryMonitor` (after `start`) |
| `get_memory_stats()` | memory tracker snapshot |
| `current_epoch` / `current_loss` / `best_loss` / `is_running` | state |

## TrainingConfig

Dataclass with validation in `__post_init__`; see
[Training](training.md#trainingconfig-reference) for the full field
tables. Construction with an invalid value raises `ValueError`.

## TelemetryMonitor

| Method | Meaning |
| --- | --- |
| `poll(max_items=4096)` | drain the C ring buffer into Python history |
| `get_latest_values()` | latest value per metric |
| `get_history_values(name)` | series for one metric |
| `compute_stats(name)` | min/max/mean/stddev/latest |
| `export_csv(path)` / `export_json(path)` | write collected history |
| `detect_anomalies()` / `detect_memory_leak()` | stream summaries |

## Low-level bindings

`cengine.bindings` mirrors the whole C ABI: `lib` (the loaded
`libcengine.so`), `check(status)` (raise `CEngineError` on non-zero),
`CeTensor` and friends, and every exported function with explicit
`argtypes`. Use it for anything the high-level API does not expose —
see `examples/03_manual_training_loop.py` for a complete optimizer loop.

Library discovery order: `CENGINE_LIB` env → package directory →
`build/lib/` → system loader. See
[Getting started](getting-started.md#build-from-source).

## Errors

| Exception | Raised when |
| --- | --- |
| `CEngineError` | a C function returns a non-zero status (with the C message) |
| `ValueError` | invalid `TrainingConfig` field |
| `OSError` | `libcengine.so` cannot be found anywhere |
