# cengine

[English](README.md) | **Русский** ([README.ru.md](README.ru.md))

[![CI](https://github.com/DKovalchuk/cengine/actions/workflows/ci.yml/badge.svg)](https://github.com/DKovalchuk/cengine/actions/workflows/ci.yml)

Tiny CPU training engine for neural networks, written in C11 (AVX2/FMA, OpenMP), with Python ctypes bindings.

## Contents

- [About](#about)
- [Features](#features)
- [Requirements](#requirements)
- [Build](#build)
- [Install with pip](#install-with-pip)
- [Documentation](#documentation)
- [Examples](#examples)
- [Testing](#testing)
- [Python usage](#python-usage)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Continuous integration](#continuous-integration)
- [Status and known limitations](#status-and-known-limitations)
- [License](#license)

## About

cengine is a self-contained training engine that runs neural network training on the CPU without external ML dependencies (no PyTorch/BLAS required). It provides tensors with an aligned memory pool, an autograd tape, layers and optimizers, a curriculum data pipeline for bit-wise ALU tasks, a threadpool with OpenMP, and telemetry — plus a thin Python layer (ctypes) for driving training from Python scripts.

## Features

- **Tensors** — aligned allocations (64-byte), memory pool, memory tracker
- **Layers** — Linear, ELU, Sigmoid, Tanh, Softmax, KAN (Kolmogorov–Arnold) layers, sequential containers, recurrent experts with carry propagation
- **Autograd** — tape-based backward pass with per-entry state destructors
- **Optimizers** — SGD, Adam, AdamW; gradient clipping; LR schedulers
- **Losses** — BCE and MSE (both mask-aware), load-balancing and Z-loss for MoE routing
- **Data** — ALU dataset generator (ADD/SUB/AND/XOR/OR), curriculum staging, prefetching dataloader with race-free epoch reseeding
- **Performance** — AVX2/FMA kernels, OpenMP batch parallelism, threadpool
- **Checkpoints** — atomic saves, validated all-or-nothing loads, PyTorch interop
- **Telemetry** — perf counters, lock-free metric ring, reporter, CSV/JSON export
- **Python bindings** — full ctypes API surface, trainer class, config dataclasses, `pip install`-able package

## Requirements

- GCC (C11) or Clang
- Linux x86-64 with AVX2 (current build; portable build without AVX2 is planned — see [Status](#status-and-known-limitations))
- Python 3.9+ with `numpy` (for the Python layer), `pytest` (for tests)

## Build

```bash
make          # shared (build/lib/libcengine.so) + static (build/lib/libcengine.a)
make clean
```

## Install with pip

```bash
pip install .    # runs `make shared` and ships libcengine.so inside the package
```

The installed package is self-contained; the library search order is
`CENGINE_LIB` env → package directory → `build/lib/` → system loader.
See [Getting started](docs/en/getting-started.md) for details.

## Documentation

Guides are available in English and Russian:

| Guide | English | Русский |
| --- | --- | --- |
| Getting started | [docs/en/getting-started.md](docs/en/getting-started.md) | [docs/ru/getting-started.md](docs/ru/getting-started.md) |
| Training (config, checkpoints, telemetry) | [docs/en/training.md](docs/en/training.md) | [docs/ru/training.md](docs/ru/training.md) |
| Python API | [docs/en/python-api.md](docs/en/python-api.md) | [docs/ru/python-api.md](docs/ru/python-api.md) |
| Architecture | [docs/en/architecture.md](docs/en/architecture.md) | [docs/ru/architecture.md](docs/ru/architecture.md) |

## Examples

Runnable scripts under [`examples/`](examples/) — every one of them is
executed by CI on each change:

- [`01_quickstart_alu.py`](examples/01_quickstart_alu.py) — train an MLP on the ALU task
- [`02_checkpoints_telemetry.py`](examples/02_checkpoints_telemetry.py) — save, reload, resume, export metrics
- [`03_manual_training_loop.py`](examples/03_manual_training_loop.py) — forward/backward/Adam without the trainer
- [`04_recurrent_expert.py`](examples/04_recurrent_expert.py) — recurrent expert with carry propagation

## Testing

```bash
# C tests
make test

# Python tests (requires `make` first)
pip install numpy pytest
python -m pytest tests/ -q
```

## Python usage

```python
from cengine import CEngineTrainer, TrainingConfig, ELU, Linear, Sequential, Sigmoid

model = Sequential(
    Linear(19, 64), ELU(),
    Linear(64, 64), ELU(),
    Linear(64, 16), Sigmoid(),
    name="alu_mlp",
)
cfg = TrainingConfig(total_epochs=10, batch_size=64, samples_per_epoch=512)
trainer = CEngineTrainer(model, cfg)
trainer.start(blocking=True)
print("final loss:", trainer.current_loss)
model.close()
```

More: [Getting started](docs/en/getting-started.md) · [Python API](docs/en/python-api.md)

## Architecture

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

Detailed layer-by-layer documentation: [Architecture](docs/en/architecture.md).

## Project structure

```
cengine/
├── include/          # Public C headers (cengine.h, tensor.h, ...)
├── src/              # Implementation (core, math, nn, autograd,
│   │                 #   training, data, parallel, telemetry)
├── cengine/          # ctypes bindings, trainer, config, telemetry
├── docs/en, docs/ru  # Bilingual guides (navigation hub: this README)
├── examples/         # Runnable usage scenarios (smoke-tested in CI)
├── tests/            # C unit tests (make test) + Python tests (pytest)
├── .github/workflows/ # CI (GitHub Actions)
├── pyproject.toml    # pip packaging (builds the C library on install)
├── Makefile
└── LICENSE           # MIT
```

## Continuous integration

Every push to `main` and every pull request runs [.github/workflows/ci.yml](.github/workflows/ci.yml):

1. `make` — build shared and static libraries
2. `make test` — C unit tests
3. `pytest` — Python tests (determinism, docs navigation, examples, packaging)
4. packaging smoke test — `pip install .` into a clean venv + import from outside the repo

`main` is protected: direct pushes and force-pushes are rejected; all changes go through pull requests.

## Status and known limitations

Early stage, hardened by an ongoing audit (see [pull requests](https://github.com/DKovalchuk/cengine/pulls)):

- **AVX2-only build** — `-march=native -mavx2` is currently required; a portable build flag is planned
- **Autograd tape** — single-entry backward works; multi-entry chains are rejected with `CE_ERR_NOT_IMPLEMENTED` — a chained tape is a planned post-release feature, and a loud refusal beats silently wrong gradients
- **`kan_recurrent`** — forward pass only; training a backward-less model fails loudly with `CE_ERR_NOT_INITIALIZED` instead of crashing
- **Test coverage** — core paths are pinned (losses, batch layout, checkpoints, threadpool, reseeding, packaging, docs navigation, examples); some C modules still lack dedicated unit tests

## License

[MIT](LICENSE)
