"""Checkpoints and telemetry: save weights, reload, resume, export metrics.

Trains briefly, saves the model to disk, loads the weights into a fresh
model (exact roundtrip check), resumes training and exports the telemetry
stream to CSV — all through the public trainer API.

Run:  python examples/02_checkpoints_telemetry.py
"""
import ctypes
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cengine import CEngineTrainer, ELU, Linear, Sequential, Sigmoid, TrainingConfig

N_BITS = 8
INPUT_DIM = 2 * N_BITS + 3
OUTPUT_DIM = 2 * N_BITS


def build_model() -> Sequential:
    return Sequential(
        Linear(INPUT_DIM, 32), ELU(),
        Linear(32, OUTPUT_DIM), Sigmoid(),
        name="alu_ckpt_demo",
    )


def make_config(workdir: str, epochs: int) -> TrainingConfig:
    return TrainingConfig(
        total_epochs=epochs,
        batch_size=64,
        samples_per_epoch=256,
        use_curriculum=False,
        min_bits=N_BITS,
        max_bits=N_BITS,
        model_input_dim=INPUT_DIM,
        model_output_dim=OUTPUT_DIM,
        checkpoint_dir=os.path.join(workdir, "auto"),
        seed=7,
    )


def first_tensor_values(model) -> list[float]:
    """Raw float values of the model's first parameter tensor."""
    p = model.parameters()[0]
    n = p.contents.numel
    arr = (ctypes.c_float * n).from_address(
        ctypes.addressof(p.contents.data.contents)
    )
    return list(arr)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cengine-ex02-") as workdir:
        saved_dir = os.path.join(workdir, "checkpoint")
        csv_path = os.path.join(workdir, "metrics.csv")

        # ── Phase 1: train and save ────────────────────────────────────────
        model = build_model()

        def on_epoch(epoch: int, loss: float, trainer: CEngineTrainer) -> None:
            # poll() drains the C ring buffer into the monitor's history —
            # export_csv only writes what poll() has collected.
            monitor = trainer.get_telemetry()
            monitor.poll()
            latest = monitor.get_latest_values()
            epoch_ms = latest.get("epoch_time_ms", "n/a")
            print(f"phase 1  epoch {epoch + 1}: loss={loss:.4f}  epoch_time={epoch_ms}")

        trainer1 = CEngineTrainer(model, make_config(workdir, 5), on_epoch_end=on_epoch)
        trainer1.start(blocking=True)
        trainer1.save_model(saved_dir)
        print("checkpoint files:", sorted(os.listdir(saved_dir)))

        # ── Roundtrip: a fresh model must reproduce the saved weights ──────
        before = first_tensor_values(model)
        fresh = build_model()
        trainer2 = CEngineTrainer(fresh, make_config(workdir, 3))
        trainer2.load_model(saved_dir)
        after = first_tensor_values(fresh)
        assert before == after, "loaded weights differ from the saved ones"
        print(f"roundtrip ok — {len(after)} values match exactly")

        # ── Phase 2: resume from loaded weights, export metrics ────────────
        phase2_epochs = 3

        def on_epoch2(epoch: int, loss: float, trainer: CEngineTrainer) -> None:
            print(f"phase 2  epoch {epoch + 1}: loss={loss:.4f}")
            monitor = trainer.get_telemetry()
            monitor.poll()
            if epoch + 1 == phase2_epochs:
                monitor.export_csv(csv_path)
                print(f"telemetry exported to {csv_path}")

        trainer2 = CEngineTrainer(fresh, make_config(workdir, phase2_epochs),
                                  on_epoch_end=on_epoch2)
        trainer2.load_model(saved_dir)
        trainer2.start(blocking=True)
        print(f"resumed run finished — loss {trainer2.current_loss:.4f}")

        assert os.path.isfile(csv_path), "telemetry CSV was not written"
        with open(csv_path, encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        print(f"csv rows: {n_lines}")

        model.close()
        fresh.close()


if __name__ == "__main__":
    main()
