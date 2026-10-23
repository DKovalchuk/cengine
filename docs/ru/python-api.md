# Python API

[English](../en/python-api.md) | [Русский](python-api.md)

**Документация:** [Начало работы](getting-started.md) · [Обучение](training.md) · Python API · [Архитектура](architecture.md)

Всё импортируется из пакета `cengine`:

```python
from cengine import (
    CEngineTrainer, TrainingConfig,
    Sequential, Linear, ELU, Sigmoid, Tanh, Softmax, GumbelSoftmax,
    RecurrentExpert, TelemetryMonitor,
    lib, check, CEngineError,
)
```

## Модули

`Module` — базовый класс; контейнеры держат детей, листья оборачивают
C-слой.

| Класс | Конструктор | Примечания |
| --- | --- | --- |
| `Sequential` | `Sequential(*layers, name="")` | цепочка слоёв; `add_module(name, module)` |
| `Linear` | `Linear(in_features, out_features, name="")` | Xavier-веса, нулевой bias |
| `ELU` | `ELU(alpha=1.0, name="")` | |
| `Sigmoid` / `Tanh` | `Sigmoid(name="")` | |
| `Softmax` | `Softmax(name="")` | backward = JVP `y·(g − Σ g·y)` |
| `GumbelSoftmax` | `GumbelSoftmax(temperature=1.0, name="")` | |
| `RecurrentExpert` | `RecurrentExpert(hidden_size=64, name=...)` | вход `[B, T, 6]`, выход `[B, T, 2]` (result, carry) |

Общее API `Module`:

| Элемент | Смысл |
| --- | --- |
| `forward(input_tensor, output_tensor)` | C-forward (ctypes-тензоры) |
| `backward(grad_output, grad_input)` | C-backward |
| `parameters()` / `gradients()`` | плоские списки указателей на тензоры (рекурсивно для контейнеров) |
| `close()` | уничтожить C-модуль; можно вызывать дважды; для детей, принадлежащих родителю, идемпотентно |
| `handle` | нижележащий `CeModule*` (исключение, если закрыт) |

Модели освобождаются автоматически при сборке мусора, но явный `close()`
(или полагаться на `close()` родительского контейнера) освобождает C-сторону
детерминированно.

## CEngineTrainer

| Элемент | Смысл |
| --- | --- |
| `CEngineTrainer(model, config, on_epoch_end=None)` | валидация конфига, подготовка C-движка |
| `start(blocking=False)` / `stop()` / `join(timeout=None)` | управление прогоном |
| `save_model(path)` / `load_model(path)` | roundtrip каталога чекпоинтов |
| `save_model_pytorch(path)` / `load_model_pytorch(path)` | интероперация с torch state-dict (нужен `torch`) |
| `get_telemetry()` | `TelemetryMonitor` (после `start`) |
| `get_memory_stats()` | снапшот трекера памяти |
| `current_epoch` / `current_loss` / `best_loss` / `is_running` | состояние |

## TrainingConfig

Dataclass с валидацией в `__post_init__`; полные таблицы полей — в
гайде [Обучение](training.md#справочник-trainingconfig). Невалидное
значение при конструировании поднимает `ValueError`.

## TelemetryMonitor

| Метод | Смысл |
| --- | --- |
| `poll(max_items=4096)` | вычитать SPSC-кольцо C в Python-историю |
| `get_latest_values()` | последнее значение каждой метрики |
| `get_history_values(name)` | ряд по одной метрике |
| `compute_stats(name)` | min/max/mean/stddev/latest |
| `export_csv(path)` / `export_json(path)` | запись накопленной истории |
| `detect_anomalies()` / `detect_memory_leak()` | сводки по потоку |

## Низкоуровневые биндинги

`cengine.bindings` зеркалит весь C ABI: `lib` (загруженный
`libcengine.so`), `check(status)` (бросает `CEngineError` при ненулевом
статусе), `CeTensor` и остальные структуры, все экспортированные функции
с явными `argtypes`. Используйте для того, что не покрыто
высокоуровневым API — полный цикл с оптимизатором в
`examples/03_manual_training_loop.py`.

Порядок поиска библиотеки: переменная `CENGINE_LIB` → каталог пакета →
`build/lib/` → системный загрузчик. См.
[Начало работы](getting-started.md#сборка-из-исходников).

## Ошибки

| Исключение | Когда |
| --- | --- |
| `CEngineError` | C-функция вернула ненулевой статус (с сообщением из C) |
| `ValueError` | невалидное поле `TrainingConfig` |
| `OSError` | `libcengine.so` не найдена нигде |
