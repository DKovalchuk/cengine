# Обучение

[English](../en/training.md) | [Русский](training.md)

**Документация:** [Начало работы](getting-started.md) · Обучение · [Python API](python-api.md) · [Архитектура](architecture.md)

Обучение задаётся двумя объектами: моделью `Module` и `TrainingConfig`,
которые оборачивает `CEngineTrainer` — он гоняет цикл в фоновом потоке,
чтобы можно было смотреть телеметрию и запрашивать остановку.

```python
from cengine import CEngineTrainer, TrainingConfig, ELU, Linear, Sequential, Sigmoid

model = Sequential(Linear(19, 64), ELU(), Linear(64, 16), Sigmoid(), name="m")
cfg = TrainingConfig(total_epochs=100, learning_rate=1e-3)

def on_epoch(epoch, loss, trainer):
    print(f"epoch {epoch}: loss={loss:.4f}")

trainer = CEngineTrainer(model, cfg, on_epoch_end=on_epoch)
trainer.start(blocking=False)   # фоновый поток
...
trainer.stop()                  # мягкая остановка после текущей эпохи
trainer.join()
```

## Жизненный цикл

| Метод | Смысл |
| --- | --- |
| `start(blocking=False)` | создаёт C-объекты и запускает цикл; `blocking=True` — синхронно |
| `stop()` | запрос мягкой остановки после текущей эпохи |
| `join(timeout=None)` | ожидание фонового потока |
| `is_running` / `current_epoch` / `current_loss` / `best_loss` | свойства состояния |
| `on_epoch_end(epoch, loss, trainer)` | колбэк после каждой эпохи (до teardown) |

NaN/inf loss честно останавливает прогон: начатая эпоха закрывается в
телеметрии и тренер печатает `stopping training`.

## Справочник TrainingConfig

У всех полей рабочие дефолты для ALU-задачи; невалидные значения поднимают
`ValueError` при конструировании (опечатка в `loss_type`/`optimizer`,
неправильный диапазон бит, нулевой батч).

**Модель / данные**

| Поле | Дефолт | Смысл |
| --- | --- | --- |
| `batch_size` | 256 | сэмплов в батче |
| `samples_per_epoch` | 2048 | генерируемых сэмплов на эпоху |
| `total_epochs` | 6000 | длина цикла |
| `min_bits` / `max_bits` | 2 / 8 | диапазон ширины операндов |
| `ops` | `[0,1,2,3]` | ADD, SUB, AND, XOR (OR = 4) |
| `use_curriculum` | `True` | стадии расширения по loss |
| `loss_threshold` / `patience` / `max_epochs_per_stage` | 0.015 / 5 / 2000 | правила перехода стадий |
| `model_input_dim` / `model_output_dim` | 0 (авто) | ширины якоря строк батча |

**Оптимизатор / loss**

| Поле | Дефолт | Смысл |
| --- | --- | --- |
| `optimizer` | `"adamw"` | `"adamw"` или `"sgd"` |
| `learning_rate` | 3e-3 | пиковый LR (OneCycle) |
| `weight_decay` | 1e-4 | регуляризация |
| `beta1` / `beta2` / `eps` | 0.9 / 0.999 / 1e-8 | моменты Adam |
| `momentum` | 0.9 | только SGD |
| `loss_type` | `"bce"` | `"bce"` или `"mse"` (оба учитывают маску батча) |
| `lambda_carry` | 0.3 | вес carry-составляющей |
| `lambda_balance` / `lambda_z` | 0.1 / 0.001 | вспомогательные loss'ы роутера MoE |
| `grad_clip_norm` | 1.0 | клиппинг градиентов |
| `pct_start` / `div_factor` / `final_div_factor` | 0.1 / 25 / 1e4 | форма OneCycle |
| `temp_start` / `temp_end` / `temp_anneal_epochs` | 2.0 / 0.5 / 2000 | отжиг температуры роутера |

**Среда выполнения**

| Поле | Дефолт | Смысл |
| --- | --- | --- |
| `num_workers` | 0 | потоки (0 = автоопределение) |
| `enable_telemetry` | `True` | поток метрик вкл/выкл |
| `telemetry_interval` | 1 | период отчёта по эпохам |
| `enable_debug_logging` | `False` | нормы градиентов на stderr по эпохам |
| `checkpoint_dir` | `"checkpoints"` | каталог авто-чекпоинтов |
| `save_every_n_epochs` | 100 | период авто-чекпоинтов (0 = выкл) |
| `seed` | 42 | поток данных + пересеивание эпох |

## Раскладка данных

Каждая строка батча якорится на ширинах модели (см.
[Архитектура](architecture.md#раскладка-батчей)):

```
input  = a_bits | b_bits | op_code(3 float, двоичный код)
target = result_bits | carry_bits
mask   = 1 на валидных позициях, 0 на паддинге
```

При `max_bits=8`: ширина input 19, target 16. Более короткие сэмплы
зануляются; маска исключает паддинг из loss. Строки якорятся на модели, а
не на батче — узкий батч не может сдвинуть колонки.

## Чекпоинты

```python
trainer.save_model("checkpoints/run1")     # во время или после прогона
other.load_model("checkpoints/run1")       # в свежесобранную модель
```

- В каталог пишутся `checkpoint_latest.bin`, `checkpoint_epoch_<N>.bin`
  и `meta.json` (эпоха, loss, конфиг).
- Сохранение атомарное: временный файл в том же каталоге пишется,
  сбрасывается и переименовывается — прерванный сейв не затирает
  предыдущий чекпоинт.
- Загрузка «всё или ничего»: форма и numel каждого параметра
  валидируются ДО записи; несовпадение поднимает `CEngineError` и не
  трогает модель.
- Sequential-контейнеры сохраняются со всеми весами (контейнер выставляет
  плоский список параметров — см. Архитектура).
- `save_model_pytorch()` / `load_model_pytorch()` доступны при
  установленном `torch`; валидируют так же, вместо «предупредить и
  оставить».

Авто-чекпоинты идут тем же путём: каждые `save_every_n_epochs` тренер
пишет `checkpoint_dir` плюс JSON-спутник на эпоху.

## Телеметрия

```python
monitor = trainer.get_telemetry()      # после start()
monitor.poll()                         # вычитать C-кольцо в историю
latest = monitor.get_latest_values()   # например {"epoch_time_ms": 4.2, ...}
monitor.export_csv("metrics.csv")      # только то, что накопил poll()
```

- `poll()` читает SPSC-кольцо; история, `get_history_values()`,
  `compute_stats()` и экспорты работают по накопленному `poll()`.
- Полезные имена метрик: `loss_total`, `epoch_time_ms`, `grad_norm`,
  `mem_peak_mb`, `curriculum_stage`, `nan_count`, `throughput`.
- `detect_anomalies()` / `detect_memory_leak()` — сводки по потоку.
- Кольцо принадлежит Python, поэтому монитор остаётся валидным после
  окончания прогона — если держите объект монитора.

## Воспроизводимость

Поток данных выводится из `seed`: эпоха `k` пересеивается `seed + k`,
поэтому два прогона с одинаковым конфигом дают одинаковые батчи. Сама
тестовая коллекция закрепляет детерминизм — ни один тест не трогает
глобальный np.random (см. `tests/test_random_determinism.py`).
