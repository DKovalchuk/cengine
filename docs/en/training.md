# Training

[English](training.md) | [Русский](../ru/training.md)

**Docs:** [Getting started](getting-started.md) · Training · [Python API](python-api.md) · [Architecture](architecture.md)

Training is driven by two objects: a `Module` model and a `TrainingConfig`,
wrapped by `CEngineTrainer`, which runs the loop in a background thread so
you can observe telemetry and request a stop.

```python
from cengine import CEngineTrainer, TrainingConfig, ELU, Linear, Sequential, Sigmoid

model = Sequential(Linear(19, 64), ELU(), Linear(64, 16), Sigmoid(), name="m")
cfg = TrainingConfig(total_epochs=100, learning_rate=1e-3)

def on_epoch(epoch, loss, trainer):
    print(f"epoch {epoch}: loss={loss:.4f}")

trainer = CEngineTrainer(model, cfg, on_epoch_end=on_epoch)
trainer.start(blocking=False)   # background thread
...
trainer.stop()                  # graceful stop after the current epoch
trainer.join()
```

## Lifecycle

| Method | Meaning |
| --- | --- |
| `start(blocking=False)` | set up C objects and run the loop; `blocking=True` runs inline |
| `stop()` | request a graceful stop after the current epoch finishes |
| `join(timeout=None)` | wait for the background thread |
| `is_running` / `current_epoch` / `current_loss` / `best_loss` | state properties |
| `on_epoch_end(epoch, loss, trainer)` | callback after every epoch (before teardown) |

NaN/inf loss stops the run honestly: the started epoch is closed in the
telemetry stream and the trainer prints `stopping training`.

## TrainingConfig reference

All fields have working defaults for the ALU task; invalid values raise
`ValueError` at construction (typoed `loss_type`, `optimizer`, bad bit
range, zero batch size).

**Model / data**

| Field | Default | Meaning |
| --- | --- | --- |
| `batch_size` | 256 | samples per batch |
| `samples_per_epoch` | 2048 | generated samples per epoch |
| `total_epochs` | 6000 | loop length |
| `min_bits` / `max_bits` | 2 / 8 | operand width range |
| `ops` | `[0,1,2,3]` | ADD, SUB, AND, XOR (OR = 4) |
| `use_curriculum` | `True` | stage widening driven by the loss |
| `loss_threshold` / `patience` / `max_epochs_per_stage` | 0.015 / 5 / 2000 | curriculum advance rules |
| `model_input_dim` / `model_output_dim` | 0 (auto) | batch-row anchoring widths |

**Optimizer / loss**

| Field | Default | Meaning |
| --- | --- | --- |
| `optimizer` | `"adamw"` | `"adamw"` or `"sgd"` |
| `learning_rate` | 3e-3 | peak LR (OneCycle) |
| `weight_decay` | 1e-4 | regularization |
| `beta1` / `beta2` / `eps` | 0.9 / 0.999 / 1e-8 | Adam moments |
| `momentum` | 0.9 | SGD only |
| `loss_type` | `"bce"` | `"bce"` or `"mse"` (both honour the batch mask) |
| `lambda_carry` | 0.3 | carry-loss weight |
| `lambda_balance` / `lambda_z` | 0.1 / 0.001 | MoE router aux losses |
| `grad_clip_norm` | 1.0 | gradient clipping |
| `pct_start` / `div_factor` / `final_div_factor` | 0.1 / 25 / 1e4 | OneCycle shape |
| `temp_start` / `temp_end` / `temp_anneal_epochs` | 2.0 / 0.5 / 2000 | router temperature annealing |

**Runtime**

| Field | Default | Meaning |
| --- | --- | --- |
| `num_workers` | 0 | threads (0 = auto) |
| `enable_telemetry` | `True` | metric stream on/off |
| `telemetry_interval` | 1 | epoch reporting period |
| `enable_debug_logging` | `False` | per-epoch gradient norms on stderr |
| `checkpoint_dir` | `"checkpoints"` | auto-checkpoint directory |
| `save_every_n_epochs` | 100 | auto-checkpoint period (0 = off) |
| `seed` | 42 | data stream + epoch reseeding |

## Data layout

Every batch row is anchored on the model's widths (see
[Architecture](architecture.md#batch-layout)):

```
input  = a_bits | b_bits | op_code(3 floats, binary-encoded)
target = result_bits | carry_bits
mask   = 1 at valid positions, 0 at padding
```

With `max_bits=8`: input width 19, target width 16. Shorter samples are
zero-padded; the mask excludes padding from the loss. Rows are re-anchored
on the model, not the batch, so a narrow batch cannot shift the columns.

## Checkpoints

```python
trainer.save_model("checkpoints/run1")     # after or during a run
other.load_model("checkpoints/run1")       # into a freshly built model
```

- Writes `checkpoint_latest.bin`, `checkpoint_epoch_<N>.bin` and
  `meta.json` (epoch, loss, config) into the directory.
- Saving is atomic: a temp file in the same directory is written, flushed
  and renamed — an interrupted save never clobbers the previous checkpoint.
- Loading is all-or-nothing: every parameter is validated (shape, numel)
  before anything is written; a mismatch raises `CEngineError` and leaves
  the model untouched.
- Sequential containers are saved with their full weights (the container
  exposes a flat parameter list — see Architecture).
- `save_model_pytorch()` / `load_model_pytorch()` are available when
  `torch` is installed; they validate the same way instead of
  warn-and-keep-going.

Automatic checkpoints use the same path: every `save_every_n_epochs` the
trainer writes `checkpoint_dir` plus a JSON sidecar per epoch.

## PyTorch interop (CUDA)

Export a checkpoint and continue training on the GPU:

```python
trainer.save_model_pytorch("model.pt")     # state_dict + architecture metadata

from cengine.pytorch_bridge import load_model_torch
torch_model = load_model_torch("model.pt") # rebuilds the equivalent torch module
torch_model.to("cuda")
# ... regular torch training ...
torch.save({"state_dict": torch_model.state_dict(),
            "metadata": {"architecture": model.architecture()}}, "model.pt")

trainer.load_model_pytorch("model.pt")     # weights back into cengine
```

- `Module.architecture()` records layer types and sizes in the `.pt`
  metadata, so the file itself says how to rebuild the model.
- Sequential checkpoints use torch `nn.Sequential` child numbering
  (`0.weight`, `2.bias`); MoE keys are `experts.{i}.{name}` —
  `load_state_dict(strict=True)` matches.
- The bridge is pinned by tests: the rebuilt torch module reproduces the
  cengine forward pass (maxdiff < 1e-5), and both loaders are two-phase
  (a bad checkpoint never partially loads).
- `RecurrentExpert` has no generic torch equivalent — build your recurrent
  module manually; its parameter keys (`w1`, `b1`, …) match the C
  registration order.
- Runnable version: [`examples/05_pytorch_cuda.py`](../../examples/05_pytorch_cuda.py).

## Telemetry

```python
monitor = trainer.get_telemetry()      # after start()
monitor.poll()                         # drain the C ring into history
latest = monitor.get_latest_values()   # e.g. {"epoch_time_ms": 4.2, ...}
monitor.export_csv("metrics.csv")      # only what poll() has collected
```

- `poll()` reads the SPSC ring buffer; history, `get_history_values()`,
  `compute_stats()` and the exports work on what `poll()` drained.
- Useful metric names: `loss_total`, `epoch_time_ms`, `grad_norm`,
  `mem_peak_mb`, `curriculum_stage`, `nan_count`, `throughput`.
- `detect_anomalies()` / `detect_memory_leak()` summarise the stream.
- The ring buffer is Python-owned, so the monitor stays valid after the
  run finishes — as long as you keep the monitor object.

## Reproducibility

The data stream is derived from `seed`: epoch `k` reseeds with
`seed + k`, so two runs with the same config produce the same batches.
The test suite itself pins determinism — no test touches the process-wide
numpy RNG (see `tests/test_random_determinism.py`).
