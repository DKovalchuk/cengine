# cengine

**English** ([README.md](README.md)) | Русский

[![CI](https://github.com/DKovalchuk/cengine/actions/workflows/ci.yml/badge.svg)](https://github.com/DKovalchuk/cengine/actions/workflows/ci.yml)

Микро-движок обучения нейросетей на CPU, написанный на C11 (AVX2/FMA, OpenMP), с Python-биндингами через ctypes.

## Содержание

- [О проекте](#о-проекте)
- [Возможности](#возможности)
- [Требования](#требования)
- [Сборка](#сборка)
- [Тесты](#тесты)
- [Использование из Python](#использование-из-python)
- [Архитектура](#архитектура)
- [Структура проекта](#структура-проекта)
- [Непрерывная интеграция](#непрерывная-интеграция)
- [Статус и известные ограничения](#статус-и-известные-ограничения)
- [Лицензия](#лицензия)

## О проекте

cengine — самодостаточный движок обучения, выполняющий тренировку нейросетей на CPU без внешних ML-зависимостей (не требует PyTorch/BLAS). Предоставляет тензоры с выровненным пулем памяти, autograd-ленту, слои и оптимизаторы, конвейер данных с курикулумом для битовых ALU-задач, threadpool с OpenMP и телеметрию — плюс тонкий Python-слой (ctypes) для управления тренировкой из скриптов.

## Возможности

- **Тензоры** — выровненные аллокации (64 байта), пул памяти, трекер памяти
- **Слои** — Linear, ELU, Sigmoid, Tanh, Softmax, слои KAN (Колмогорова–Арнольда), контейнеры sequential, рекуррентные эксперты с carry-пропагацией
- **Autograd** — backward-проход на основе ленты (tape)
- **Оптимизаторы** — SGD, Adam, AdamW; клиппинг градиентов; LR-планировщики
- **Функции потерь** — BCE, MSE, load-balancing и Z-loss для маршрутизации MoE
- **Данные** — генератор ALU-датасета (ADD/SUB/AND/XOR/OR), стадии курикулума, dataloader
- **Производительность** — AVX2/FMA-ядра, параллелизм по батчу через OpenMP, threadpool
- **Телеметрия** — perf-счётчики, коллектор метрик, репортёр
- **Python-биндинги** — полный ctypes-API (172 функции), класс тренера, dataclass-конфиги

## Требования

- GCC (C11) или Clang
- Linux x86-64 с AVX2 (текущая сборка; переносимая сборка без AVX2 в планах — см. [Статус](#статус-и-известные-ограничения))
- Python 3.9+ с `numpy` (для Python-слоя), `pytest` (для тестов)

## Сборка

```bash
make          # разделяемая (build/lib/libcengine.so) + статическая (build/lib/libcengine.a)
make clean
```

## Тесты

```bash
# C-тесты
make test

# Python-тесты (сначала выполнить `make`)
pip install numpy pytest
python -m pytest tests/ -q
```

## Использование из Python

```python
import sys
sys.path.insert(0, ".")

from python.bindings import lib, check
from python.module import RecurrentExpert

check(lib.ce_init())

expert = RecurrentExpert(hidden_size=8, name="demo")
# тензоры, forward/backward — см. примеры в tests/test_recurrent_cengine.py
```

## Архитектура

```
Python (ctypes)                    Ядро C11
─────────────────                  ─────────────────────────────
python/bindings.py   ── FFI ──►    core/       тензор, пул памяти
python/module.py                 │   math/      AVX2-ядра
python/trainer.py                │   nn/        слои, рекуррент
python/config.py                 │   autograd/  лента, backward
python/telemetry.py              │   training/  тренер, оптимизаторы
                                 │   data/      генератор ALU, курикулум
                                 │   parallel/  threadpool, барьеры
                                 │   telemetry/ счётчики, репортёр
```

Подробная документация по слоям: [docs/](docs/).

## Структура проекта

```
cengine/
├── include/          # Публичные C-заголовки (cengine.h, tensor.h, ...)
├── src/              # Реализация (core, math, nn, autograd,
│   │                 #   training, data, parallel, telemetry)
├── python/           # ctypes-биндинги, тренер, конфиг, телеметрия
├── tests/
│   ├── test_phase1.c             # C-тесты (make test)
│   └── test_recurrent_cengine.py # Python-тесты (pytest)
├── .github/workflows/            # CI (GitHub Actions)
├── Makefile
└── LICENSE           # MIT
```

## Непрерывная интеграция

Каждый пуш в `main` и каждый pull request запускают [.github/workflows/ci.yml](.github/workflows/ci.yml):

1. `make` — сборка разделяемой и статической библиотек
2. `make test` — C-тесты
3. `pytest` — Python-тесты

`main` защищён: прямые пуши и force-push отклоняются; все изменения идут через pull request.

## Статус и известные ограничения

Ранняя стадия, идёт активная доводка (актуальную работу см. в [pull requests](https://github.com/DKovalchuk/cengine/pulls)):

- **Валидация формы входа** — исправляется баг, из-за которого рекуррентный forward предполагает жёстко заданную ширину входа; до этого Python-тесты могут падать непостоянно (отслеживается в CI)
- **Сборка только под AVX2** — сейчас требуется `-march=native`; переносимый флаг сборки в планах
- **Известные пробелы** — tape-backward считает только последний узел; у `kan_recurrent` нет backward-прохода; некоторые C-модули ещё не покрыты тестами

## Лицензия

[MIT](LICENSE) © Dmitry
