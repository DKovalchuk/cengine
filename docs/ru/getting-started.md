# Начало работы

[English](../en/getting-started.md) | [Русский](getting-started.md)

**Документация:** Начало работы · [Обучение](training.md) · [Python API](python-api.md) · [Архитектура](architecture.md)

cengine — самодостаточный движок обучения нейросетей на CPU, написанный на
C11 (AVX2/FMA, OpenMP) с Python-биндингами через ctypes. Внешних
ML-зависимостей нет — ни PyTorch, ни BLAS: всё лежит в этом репозитории.

## Требования

- GCC (C11) или Clang
- Linux x86-64 с поддержкой AVX2
- Python 3.9+ с `numpy` (для Python-слоя), `pytest` (для тестов)
- `make`

## Сборка из исходников

```bash
make          # build/lib/libcengine.so (разделяемая) + libcengine.a (статическая)
make test     # собрать и запустить C-тесты
```

## Установка через pip

```bash
pip install .             # запускает `make shared` и кладёт libcengine.so внутрь пакета
```

Установленный пакет самодостаточен: скомпилированная библиотека ложится
рядом с Python-модулями, и ctypes находит её автоматически.

```bash
pip install git+https://github.com/DKovalchuk/cengine.git
```

В dev-чекауте pip не обязателен — биндинги найдут
`build/lib/libcengine.so` из корня репозитория. Порядок поиска:

1. переменная окружения `CENGINE_LIB` (явный оверрайд);
2. `libcengine.so` рядом с установленным пакетом;
3. `build/lib/libcengine.so` (репозиторий после `make`);
4. системный загрузчик (`LD_LIBRARY_PATH`).

## Первый запуск обучения

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

19 = 8 бит `a` + 8 бит `b` + 3-float код операции; 16 = 8 бит результата +
8 бит carry. Полное описание раскладки — в гайде
[Обучение](training.md).

Рабочие версии этого и ещё трёх сценариев лежат в
[`examples/`](../../examples/):

- [`01_quickstart_alu.py`](../../examples/01_quickstart_alu.py) — обучение MLP на ALU-задаче
- [`02_checkpoints_telemetry.py`](../../examples/02_checkpoints_telemetry.py) — сохранение, перезагрузка, resume, экспорт метрик
- [`03_manual_training_loop.py`](../../examples/03_manual_training_loop.py) — forward/backward/Adam без тренера
- [`04_recurrent_expert.py`](../../examples/04_recurrent_expert.py) — рекуррентный эксперт с carry-пропагацией

## Запуск тестов

```bash
make test                   # C-тесты
python -m pytest tests/ -q   # Python-тесты (сначала `make`)
```

Непрерывная интеграция (GitHub Actions) запускает те же три шага на каждом
push и pull request: сборка, C-тесты, Python-тесты — плюс smoke-тест
пакетирования (установка в чистый venv и импорт извне репозитория).

## Решение проблем

**`Cannot load libcengine.so`** — сначала соберите `make` или укажите
путь к библиотеке в `CENGINE_LIB`.

**`illegal instruction` при запуске** — сборка использует
`-march=native -mavx2`; пересоберите на машине с AVX2. Переносимая сборка
в планах (см. [Архитектура](architecture.md)).

**`pip` отказывается ставить в системный Python** (PEP 668) — сначала
создайте venv: `python3 -m venv .venv && .venv/bin/pip install .`
