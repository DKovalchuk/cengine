# cengine

[English](README.md) | **Русский** ([README.ru.md](README.ru.md))

[![CI](https://github.com/DKovalchuk/cengine/actions/workflows/ci.yml/badge.svg)](https://github.com/DKovalchuk/cengine/actions/workflows/ci.yml)

Tiny CPU training engine for neural networks, written in C11 (AVX2/FMA, OpenMP), with Python ctypes bindings.

## Contents

- [About](#about)
- [Features](#features)
- [Requirements](#requirements)
- [Build](#build)
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
- **Autograd** — tape-based backward pass
- **Optimizers** — SGD, Adam, AdamW; gradient clipping; LR schedulers
- **Losses** — BCE, MSE, load-balancing and Z-loss for MoE routing
- **Data** — ALU dataset generator (ADD/SUB/AND/XOR/OR), curriculum staging, dataloader
- **Performance** — AVX2/FMA kernels, OpenMP batch parallelism, threadpool
- **Telemetry** — perf counters, metric collector, reporter
- **Python bindings** — full ctypes API surface (172 functions), trainer class, config dataclasses

## Requirements

- GCC (C11) or Clang
- Linux x86-64 with AVX2 (current build; portable build without AVX2 is planned — see [Status](#status-and-known-limitations))
- Python 3.9+ with `numpy` (for the Python layer), `pytest` (for tests)

## Build

```bash
make          # shared (build/lib/libcengine.so) + static (build/lib/libcengine.a)
make clean
```

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
import sys
sys.path.insert(0, ".")

from python.bindings import lib, check
from python.module import RecurrentExpert

check(lib.ce_init())

expert = RecurrentExpert(hidden_size=8, name="demo")
# tensors, forward/backward — see tests/test_recurrent_cengine.py for examples
```

## Architecture

```
Python (ctypes)                    C11 core
─────────────────                  ─────────────────────────────
python/bindings.py   ── FFI ──►    core/       tensor, memory pool
python/module.py                 │   math/      AVX2 kernels
python/trainer.py                │   nn/        layers, recurrent
python/config.py                 │   autograd/  tape, backward
python/telemetry.py              │   training/  trainer, optimizers
                                 │   data/      ALU gen, curriculum
                                 │   parallel/  threadpool, barriers
                                 │   telemetry/ counters, reporter
```

Detailed layer-by-layer documentation: [docs/](docs/).

## Project structure

```
cengine/
├── include/          # Public C headers (cengine.h, tensor.h, ...)
├── src/              # Implementation (core, math, nn, autograd,
│   │                 #   training, data, parallel, telemetry)
├── python/           # ctypes bindings, trainer, config, telemetry
├── tests/
│   ├── test_phase1.c             # C unit tests (make test)
│   └── test_recurrent_cengine.py # Python tests (pytest)
├── .github/workflows/            # CI (GitHub Actions)
├── Makefile
└── LICENSE           # MIT
```

## Continuous integration

Every push to `main` and every pull request runs [.github/workflows/ci.yml](.github/workflows/ci.yml):

1. `make` — build shared and static libraries
2. `make test` — C unit tests
3. `pytest` — Python tests

`main` is protected: direct pushes and force-pushes are rejected; all changes go through pull requests.

## Status and known limitations

Early stage, under active hardening (see [pull requests](https://github.com/DKovalchuk/cengine/pulls) for the current work):

- **Input shape validation** — a bug where the recurrent forward pass assumes a hardcoded input width is being fixed; until then Python tests may fail intermittently (tracked in CI)
- **AVX2-only build** — `-march=native` is currently required; a portable build flag is planned
- **Known gaps** — tape backward handles only the last node; `kan_recurrent` has no backward pass; some C modules are not yet covered by tests

## License

[MIT](LICENSE)
