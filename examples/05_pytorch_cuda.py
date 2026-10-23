"""PyTorch interop: continue training on CUDA through the bridge.

Trains briefly with cengine (CPU), exports a .pt with architecture
metadata, rebuilds an equivalent torch module, takes torch optimizer
steps (on CUDA when available), and hands the weights back to cengine.

Run:  python examples/05_pytorch_cuda.py
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    print("torch is not installed - skipping the PyTorch bridge demo")
    sys.exit(0)

import numpy as np

from cengine import CEngineTrainer, ELU, Linear, Sequential, Sigmoid, TrainingConfig
from cengine.pytorch_bridge import load_model_torch

N_BITS = 8
INPUT_DIM = 2 * N_BITS + 3   # a_bits + b_bits + op_code
OUTPUT_DIM = 2 * N_BITS       # result_bits + carry_bits


def build_model() -> Sequential:
    return Sequential(
        Linear(INPUT_DIM, 32), ELU(),
        Linear(32, OUTPUT_DIM), Sigmoid(),
        name="bridge_alu",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cengine-ex05-") as workdir:
        pt_path = str(pathlib.Path(workdir) / "model.pt")
        back_path = str(pathlib.Path(workdir) / "model_back.pt")
        cfg = TrainingConfig(
            total_epochs=3,
            batch_size=64,
            samples_per_epoch=256,
            use_curriculum=False,
            min_bits=N_BITS,
            max_bits=N_BITS,
            model_input_dim=INPUT_DIM,
            model_output_dim=OUTPUT_DIM,
            checkpoint_dir=str(pathlib.Path(workdir) / "ckpt"),
            seed=11,
        )

        # ── 1. cengine trains on the CPU ────────────────────────────────
        model = build_model()
        trainer = CEngineTrainer(model, cfg)
        trainer.start(blocking=True)
        print(f"cengine phase: loss {trainer.current_loss:.4f}")

        # ── 2. export .pt, rebuild the same model in torch ──────────────
        trainer.save_model_pytorch(pt_path)
        torch_model = load_model_torch(pt_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_model.to(device)
        torch_model.train()
        n_params = sum(p.numel() for p in torch_model.parameters())
        print(f"torch phase: {n_params} params on {device}")

        # ── 3. torch optimizer steps (GPU when available) ───────────────
        rng = np.random.default_rng(5)
        x_np = rng.uniform(-1.0, 1.0, (128, INPUT_DIM)).astype(np.float32)
        x = torch.from_numpy(x_np).to(device)
        # Toy target: echo the operand-bit columns into both target halves.
        y = torch.zeros(128, OUTPUT_DIM, device=device)
        y[:, :N_BITS] = x[:, :N_BITS]
        y[:, N_BITS:2 * N_BITS] = x[:, N_BITS:2 * N_BITS]

        opt = torch.optim.AdamW(torch_model.parameters(), lr=1e-3)
        loss = None
        for _ in range(50):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(torch_model(x), y)
            loss.backward()
            opt.step()
        print(f"torch steps: loss {loss.item():.4f}")

        # ── 4. hand the weights back to cengine ─────────────────────────
        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu() for k, v in torch_model.state_dict().items()
                },
                "metadata": {"architecture": model.architecture()},
            },
            back_path,
        )

        resumed = build_model()
        resumed_trainer = CEngineTrainer(resumed, cfg)
        resumed_trainer.load_model_pytorch(back_path)
        print("weights handed back to cengine - round trip complete")

        model.close()
        resumed.close()


if __name__ == "__main__":
    main()
