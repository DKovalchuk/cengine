# cengine

Tiny CPU training engine for neural networks written in C11 (AVX2/FMA, OpenMP), with Python ctypes bindings.

Features: tensors with memory pool, autograd tape, layers (Linear, ELU, Sigmoid, Tanh, KAN, recurrent experts), optimizers (SGD/Adam/AdamW), curriculum data pipeline, threadpool, telemetry.

## Build

```bash
make          # shared + static libraries into build/lib
make test     # build and run C tests
```

## Python

```bash
pip install numpy pytest
python -m pytest tests/ -q
```

## Status

Early stage. See [LICENSE](LICENSE) (MIT).
