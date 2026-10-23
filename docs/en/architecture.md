# Architecture

[English](architecture.md) | [Русский](../ru/architecture.md)

**Docs:** [Getting started](getting-started.md) · [Training](training.md) · [Python API](python-api.md) · Architecture

## Layer map

```
Python (ctypes)                    C11 core
─────────────────                  ─────────────────────────────
cengine/bindings.py  ── FFI ──►    core/       tensor, memory pool
cengine/module.py                │   math/      AVX2 kernels
cengine/trainer.py               │   nn/        layers, recurrent
cengine/config.py                │   autograd/  tape, backward
cengine/telemetry.py             │   training/  trainer, optimizers
                                 │   data/      ALU gen, curriculum
                                 │   parallel/  threadpool, barriers
                                 │   telemetry/ counters, reporter
```

`python/bindings.py` is the single source of truth for the FFI boundary:
every public C struct is mirrored as a `ctypes.Structure` with explicit
padding, and every function gets explicit `argtypes`/`restype` (a missing
64-bit `argtype` silently truncates seeds at the boundary — the tests pin
this).

## Tensors and memory

- 64-byte aligned allocations through a central pool, with a memory tracker
  (`ce_tracker_get_stats` feeds the `mem_*` telemetry metrics).
- `refcount` on every tensor: `ce_tensor_list_append` retains, list
  destruction releases. Containers rely on this — a `Sequential` exposes
  its children's tensors on its own parameter list (retained), and those
  lists are destroyed after the children, keeping ownership balanced.

## Layers and containers

Each layer implements `forward` / `backward` function pointers on
`ce_module_t`. The `Sequential` container chains children in order for
forward and in reverse for backward, allocating intermediate buffers that
follow each child's output/input width (a mid-chain KAN layer used to
receive an input-sized buffer — the widths now come from the layer
struct).

Parameter access is two-fold:

- **Leaf modules** own their `parameters` / `gradients` lists.
- **Containers** expose a flat parent list registered by
  `ce_sequential_add`, so generic consumers of `ce_module_param_count`
  (checkpoint save/load, the C trainer) see the whole model. Python's
  `Module._collect_params` reads parent lists only for leaves, so
  consumers never see duplicates.

## Autograd tape

The tape records `(output, grad, backward_fn, state, state_free)` entries
and replays them in reverse. Layer states are freed through the
`state_free` destructor recorded per entry (otherwise retained tensors
leak after `clear`/`destroy`). Single-entry and empty tapes are supported;
**multi-entry chains are rejected with `CE_ERR_NOT_IMPLEMENTED`** — the
full chained tape is a planned post-release feature, and refusing beats
silently computing wrong gradients.

## Data pipeline

```
ALU generator ──► dataloader prefetch worker ──► ready batch
      ▲                    (pthread)
      └── ce_dataloader_reseed() joins the worker, applies the new
          seed/op filter, discards stale batches, restarts prefetch
```

- The generator produces bit-wise ALU samples (ADD/SUB/AND/XOR/OR) with a
  3-float binary `op_code` (`op & 1`, `(op >> 1) & 1`, `(op >> 2) & 1`).
- Curriculum staging widens the bit range and op set as the loss drops.
- Epoch boundaries reseed the stream for shuffling. Because the generator
  is shared with the prefetch thread, every reconfiguration goes through
  `ce_dataloader_reseed()`, which joins the worker first — direct field
  writes were a data race (pinned by a test that scans the trainer source).

## Batch layout

Rows are anchored on the **model's** dimensions (mirroring C
`ce_trainer_prepare_batch`):

```
input  = [a_bits(0..n) | b_bits(anchor..) | op_code(2*anchor..+3)]
             anchor  = (max_input_dim - 3) // 2
target = [result_bits(0..n) | carry_bits(t_anchor..)]
             t_anchor = max_target_dim // 2
mask   = 1.0 at valid result/carry positions, 0.0 at padding
```

Anchoring on the batch width instead shifted columns whenever the batch was
narrower than the model (op code landed in b-bit columns); that regression
has a dedicated test.

## Threading

- **OpenMP** for batch-parallel kernels; `-fopenmp` at build time.
- **Prefetch worker** (one pthread per dataloader) keeps data generation off
  the training loop.
- **Threadpool** for task graphs: shutdown sets the flag and broadcasts
  under the queue lock — a lost wakeup there was a race (proved and fixed
  under ThreadSanitizer with a standalone `-fsanitize=thread` build).

## Telemetry

```
collector ──► SPSC ring buffer (4096 × 48-byte records) ──► TelemetryMonitor
   │            (epoch/forward/backward/optimizer timing,           │
   │             loss, grad norm, memory, curriculum stage …)        └── poll() drains
   └── reporter builds flat snapshots                                   into Python history
```

The ring is Python-owned (a ctypes struct) so `poll()` stays valid after
trainer teardown; `export_csv()`/`export_json()` write only what `poll()`
has collected.

## Build and CI

- `Makefile` auto-discovers sources; `-O3 -march=native -mavx2 -mfma`
  (a portable no-AVX2 build is planned).
- CI (GitHub Actions): `make` → `make test` → `pytest tests/ -q` → packaging
  smoke test (`pip install .` into a clean venv + import from outside the
  repo).
- The test suite pins the cross-cutting properties: determinism (no global
  RNG use), documentation navigation (every local link resolves), and every
  example running end-to-end.
