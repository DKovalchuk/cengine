"""Manual training loop without the trainer.

Drives the ctypes API directly: build tensors, run forward/backward
through a small MLP, compute MSE and take Adam steps — watching the
loss fall.

Run:  python examples/03_manual_training_loop.py
"""
import ctypes
import pathlib
import sys
from ctypes import POINTER, c_float, c_int

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from cengine import ELU, Linear, Sequential
from cengine.bindings import CeTensor, lib, check

BATCH, IN, HID, OUT = 64, 4, 32, 1
STEPS = 500


def make_tensor(shape, values=None):
    """Allocate a 2D float32 tensor, optionally filled from a numpy array."""
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)
    t = lib.ce_tensor_zeros(shape_arr, len(shape))
    if not t:
        raise RuntimeError("ce_tensor_zeros returned NULL")
    if values is not None:
        flat = np.asarray(values, dtype=np.float32).ravel()
        for i, v in enumerate(flat):
            t.contents.data[i] = float(v)
    return t


def main() -> None:
    # Learn y = (x0 + x1 - x2 + x3) / 4 from random inputs.
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(BATCH, IN))
    y = ((x[:, 0] + x[:, 1] - x[:, 2] + x[:, 3]) / 4.0)[:, None]

    x_t = make_tensor((BATCH, IN), x)
    y_t = make_tensor((BATCH, OUT), y)
    out_t = make_tensor((BATCH, OUT))
    g_out = make_tensor((BATCH, OUT))
    g_in = make_tensor((BATCH, IN))

    model = Sequential(Linear(IN, HID), ELU(), Linear(HID, OUT), name="manual_mlp")

    # Optimizer bound to the model's parameter/gradient tensors.
    params = model.parameters()
    grads = model.gradients()
    P = (POINTER(CeTensor) * len(params))(*params)
    G = (POINTER(CeTensor) * len(grads))(*grads)
    opt = lib.ce_adam_create(
        c_float(1e-2), c_float(0.9), c_float(0.999),
        c_float(1e-8), c_float(0.0),
        P, G, c_int(len(params)),
    )
    if not opt:
        raise RuntimeError("ce_adam_create returned NULL")

    def one_step() -> float:
        lib.ce_module_default_zero_grad(model.handle)
        model.forward(x_t, out_t)
        loss = float(lib.ce_mse_loss(out_t, y_t, None))
        check(lib.ce_mse_backward(out_t, y_t, None, g_out))
        model.backward(g_out, g_in)
        check(lib.ce_adam_step(opt))
        return loss

    first_loss = None
    last_loss = None
    for step in range(STEPS):
        loss = one_step()
        if first_loss is None:
            first_loss = loss
        if step % 100 == 0:
            print(f"step {step:3d}  mse={loss:.6f}")
        last_loss = loss

    print(f"first mse={first_loss:.6f} -> final mse={last_loss:.6f}")
    assert last_loss < first_loss, "training did not improve the loss"

    lib.ce_adam_destroy(opt)
    model.close()
    print("ok")


if __name__ == "__main__":
    main()
