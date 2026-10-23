# Getting started

[English](getting-started.md) | [Русский](../ru/getting-started.md)

**Docs:** Getting started · [Training](training.md) · [Python API](python-api.md) · [Architecture](architecture.md)

cengine is a self-contained CPU training engine for neural networks, written
in C11 (AVX2/FMA, OpenMP) with Python bindings via ctypes. It has no external
ML dependencies — no PyTorch, no BLAS — everything ships in this repository.

## Requirements

- GCC (C11) or Clang
- Linux x86-64 with AVX2 support
- Python 3.9+ with `numpy` (the Python layer), `pytest` (tests)
- `make`

## Build from source

```bash
make          # build/lib/libcengine.so (shared) + libcengine.a (static)
make test     # build and run the C unit tests
```

## Install with pip

```bash
pip install .             # runs `make shared` and ships libcengine.so inside the package
```

The installed package is self-contained: the compiled library lands next to
the Python modules and ctypes finds it automatically.

```bash
pip install git+https://github.com/DKovalchuk/cengine.git
```

For a development checkout you can skip pip entirely — the bindings find
`build/lib/libcengine.so` from the repository root. The search order is:

1. `CENGINE_LIB` environment variable (explicit override),
2. `libcengine.so` next to the installed package,
3. `build/lib/libcengine.so` (repository after `make`),
4. the system loader (`LD_LIBRARY_PATH`).

## First training run

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

19 = 8 bits of `a` + 8 bits of `b` + a 3-float operation code; 16 = 8 result
bits + 8 carry bits. See [Training](training.md) for the full data layout.

Runnable versions of this and three more scenarios live in
[`examples/`](../../examples/):

- [`01_quickstart_alu.py`](../../examples/01_quickstart_alu.py) — train an MLP on the ALU task
- [`02_checkpoints_telemetry.py`](../../examples/02_checkpoints_telemetry.py) — save, reload, resume, export metrics
- [`03_manual_training_loop.py`](../../examples/03_manual_training_loop.py) — forward/backward/Adam without the trainer
- [`04_recurrent_expert.py`](../../examples/04_recurrent_expert.py) — recurrent expert with carry propagation

## Running tests

```bash
make test               # C unit tests
python -m pytest tests/ -q   # Python tests (requires `make` first)
```

Continuous integration (GitHub Actions) runs the same three steps on every
push and pull request: build, C tests, Python tests, plus a packaging smoke
test that installs the package into a clean virtualenv.

## Troubleshooting

**`Cannot load libcengine.so`** — build it first with `make`, or point
`CENGINE_LIB` at an existing library file.

**`illegal instruction` at startup** — the build uses `-march=native -mavx2`;
rebuild on a machine with AVX2. A portable build is planned (see
[Architecture](architecture.md)).

**`pip` refuses to install into the system Python** (PEP 668) — create a
virtualenv first: `python3 -m venv .venv && .venv/bin/pip install .`
