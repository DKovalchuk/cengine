# cengine

**English** ([README.md](README.md)) | Русский

[![CI](https://github.com/DKovalchuk/cengine/actions/workflows/ci.yml/badge.svg)](https://github.com/DKovalchuk/cengine/actions/workflows/ci.yml)

Микро-движок обучения нейросетей на CPU, написанный на C11 (AVX2/FMA, OpenMP), с Python-биндингами через ctypes.

## Содержание

- [О проекте](#о-проекте)
- [Возможности](#возможности)
- [Требования](#требования)
- [Сборка](#сборка)
- [Установка через pip](#установка-через-pip)
- [Документация](#документация)
- [Примеры](#примеры)
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
- **Autograd** — backward-проход на основе ленты с деструкторами состояний
- **Оптимизаторы** — SGD, Adam, AdamW; клиппинг градиентов; LR-планировщики
- **Функции потерь** — BCE и MSE (обе учитывают маску), load-balancing и Z-loss для маршрутизации MoE
- **Данные** — генератор ALU-датасета (ADD/SUB/AND/XOR/OR), стадии курикулума, prefetch-даталоадер с гонко-безопасным пересеиванием эпох
- **Производительность** — AVX2/FMA-ядра, параллелизм по батчу через OpenMP, threadpool
- **Чекпоинты** — атомарные сейвы, валидируемые all-or-nothing загрузки, интероперация с PyTorch
- **Телеметрия** — perf-счётчики, lock-free кольцо метрик, репортёр, экспорт CSV/JSON
- **Python-биндинги** — полный ctypes-API, класс тренера, dataclass-конфиги, устанавливаемый через `pip install` пакет

## Требования

- GCC (C11) или Clang
- Linux x86-64 с AVX2 (текущая сборка; переносимая сборка без AVX2 в планах — см. [Статус](#статус-и-известные-ограничения))
- Python 3.9+ с `numpy` (для Python-слоя), `pytest` (для тестов)

## Сборка

```bash
make          # разделяемая (build/lib/libcengine.so) + статическая (build/lib/libcengine.a)
make clean
```

## Установка через pip

```bash
pip install .    # запускает `make shared` и кладёт libcengine.so внутрь пакета
```

Установленный пакет самодостаточен; порядок поиска библиотеки:
переменная `CENGINE_LIB` → каталог пакета → `build/lib/` → системный
загрузчик. Подробности — в гайде
[Начало работы](docs/ru/getting-started.md).

## Документация

Гайды на английском и русском:

| Гайд | English | Русский |
| --- | --- | --- |
| Начало работы | [docs/en/getting-started.md](docs/en/getting-started.md) | [docs/ru/getting-started.md](docs/ru/getting-started.md) |
| Обучение (конфиг, чекпоинты, телеметрия) | [docs/en/training.md](docs/en/training.md) | [docs/ru/training.md](docs/ru/training.md) |
| Python API | [docs/en/python-api.md](docs/en/python-api.md) | [docs/ru/python-api.md](docs/ru/python-api.md) |
| Архитектура | [docs/en/architecture.md](docs/en/architecture.md) | [docs/ru/architecture.md](docs/ru/architecture.md) |

## Примеры

Запускаемые скрипты в [`examples/`](examples/) — каждый из них CI
выполняет на каждом изменении:

- [`01_quickstart_alu.py`](examples/01_quickstart_alu.py) — обучение MLP на ALU-задаче
- [`02_checkpoints_telemetry.py`](examples/02_checkpoints_telemetry.py) — сохранение, перезагрузка, resume, экспорт метрик
- [`03_manual_training_loop.py`](examples/03_manual_training_loop.py) — forward/backward/Adam без тренера
- [`04_recurrent_expert.py`](examples/04_recurrent_expert.py) — рекуррентный эксперт с carry-пропагацией

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

Больше: [Начало работы](docs/ru/getting-started.md) · [Python API](docs/ru/python-api.md)

## Архитектура

```
Python (ctypes)                    Ядро C11
─────────────────                  ─────────────────────────────
cengine/bindings.py  ── FFI ──►    core/       тензор, пул памяти
cengine/module.py                │   math/      AVX2-ядра
cengine/trainer.py               │   nn/        слои, рекуррент
cengine/config.py                │   autograd/  лента, backward
cengine/telemetry.py             │   training/  тренер, оптимизаторы
                                 │   data/      генератор ALU, курикулум
                                 │   parallel/  threadpool, барьеры
                                 │   telemetry/ счётчики, репортёр
```

Подробная документация по слоям: [Архитектура](docs/ru/architecture.md).

## Структура проекта

```
cengine/
├── include/          # Публичные C-заголовки (cengine.h, tensor.h, ...)
├── src/              # Реализация (core, math, nn, autograd,
│   │                 #   training, data, parallel, telemetry)
├── cengine/          # ctypes-биндинги, тренер, конфиг, телеметрия
├── docs/en, docs/ru  # Двуязычные гайды (hub навигации — этот README)
├── examples/         # Запускаемые сценарии использования (smoke в CI)
├── tests/            # C-тесты (make test) + Python-тесты (pytest)
├── .github/workflows/ # CI (GitHub Actions)
├── pyproject.toml    # pip-пакетирование (сборка C при установке)
├── Makefile
└── LICENSE           # MIT
```

## Непрерывная интеграция

Каждый пуш в `main` и каждый pull request запускают [.github/workflows/ci.yml](.github/workflows/ci.yml):

1. `make` — сборка разделяемой и статической библиотек
2. `make test` — C-тесты
3. `pytest` — Python-тесты (детерминизм, навигация доков, примеры, пакетирование)
4. smoke-тест пакетирования — `pip install .` в чистый venv + импорт извне репозитория

`main` защищён: прямые пуши и force-push отклоняются; все изменения идут через pull request.

## Статус и известные ограничения

Ранняя стадия, идёт активная доводка (актуальную работу см. в [pull requests](https://github.com/DKovalchuk/cengine/pulls)):

- **Сборка только под AVX2** — сейчас требуется `-march=native -mavx2`; переносимый флаг сборки в планах
- **Autograd-лента** — одиночная backward-запись работает; мульти-записные цепочки отклоняются с `CE_ERR_NOT_IMPLEMENTED` — связанная лента отложена как пост-релизная фича, а громкий отказ лучше молча неверных градиентов
- **`kan_recurrent`** — только forward; попытка обучить модель без backward честно падает с `CE_ERR_NOT_INITIALIZED` вместо краха
- **Покрытие тестами** — ядерные пути закреплены (loss'ы, раскладка батчей, чекпоинты, threadpool, пересеивание, пакетирование, навигация доков, примеры); у некоторых C-модулей пока нет отдельных тестов

## Лицензия

[MIT](LICENSE)
