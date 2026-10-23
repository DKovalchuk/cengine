"""Quickstart: train a small MLP on the ALU bit task.

Builds a Sequential model and trains it with CEngineTrainer on 8-bit
ADD/SUB/AND/XOR samples, printing the loss after every epoch.

Run:  python examples/01_quickstart_alu.py
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cengine import CEngineTrainer, ELU, Linear, Sequential, Sigmoid, TrainingConfig

N_BITS = 8
INPUT_DIM = 2 * N_BITS + 3   # a_bits(8) + b_bits(8) + op_code(3)
OUTPUT_DIM = 2 * N_BITS       # result_bits(8) + carry_bits(8)


def build_model() -> Sequential:
    return Sequential(
        Linear(INPUT_DIM, 64), ELU(),
        Linear(64, 64), ELU(),
        Linear(64, OUTPUT_DIM), Sigmoid(),
        name="alu_mlp",
    )


def main() -> None:
    model = build_model()

    # Automatic checkpoints go to a temp dir — nothing is left behind.
    with tempfile.TemporaryDirectory(prefix="cengine-ex01-") as workdir:
        cfg = TrainingConfig(
            total_epochs=10,
            batch_size=64,
            samples_per_epoch=512,
            learning_rate=1e-3,
            use_curriculum=False,
            min_bits=N_BITS,
            max_bits=N_BITS,
            model_input_dim=INPUT_DIM,
            model_output_dim=OUTPUT_DIM,
            checkpoint_dir=workdir,
            seed=42,
        )

        def on_epoch(epoch: int, loss: float, trainer: CEngineTrainer) -> None:
            print(f"epoch {epoch + 1:3d}/{cfg.total_epochs}  loss={loss:.4f}")

        trainer = CEngineTrainer(model, cfg, on_epoch_end=on_epoch)
        trainer.start(blocking=True)

    print(f"done — final loss {trainer.current_loss:.4f}")
    model.close()


if __name__ == "__main__":
    main()
