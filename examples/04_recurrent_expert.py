"""Recurrent expert: forward/backward on bit sequences with carry.

Feeds a batch of 6-bit ADD sequences through RecurrentExpert — input
features per timestep are [a_bit, b_bit, carry_slot, op(3)] — and shows
the untrained predictions against the ground-truth result bits.

Run:  python examples/04_recurrent_expert.py
"""
import ctypes
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from cengine import RecurrentExpert
from cengine.bindings import lib, check

BATCH, N_BITS, HIDDEN = 8, 6, 16
# op_code is a 3-float binary encoding; op 0 = ADD -> (0, 0, 0).
OP_ADD = (0.0, 0.0, 0.0)


def make_tensor_3d(shape, values=None):
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
    rng = np.random.default_rng(4)
    a = rng.integers(0, 1 << N_BITS, size=BATCH)
    b = rng.integers(0, 1 << N_BITS, size=BATCH)
    total = a + b

    # Sequence layout: timestep t carries bit t (LSB first).
    x = np.zeros((BATCH, N_BITS, 6), dtype=np.float32)
    result_bits = np.zeros((BATCH, N_BITS), dtype=np.float32)
    for t in range(N_BITS):
        x[:, t, 0] = (a >> t) & 1
        x[:, t, 1] = (b >> t) & 1
        # x[:, t, 2] — carry slot placeholder, x[:, t, 3:6] — op_code
        x[:, t, 3:6] = OP_ADD
        result_bits[:, t] = (total >> t) & 1

    x_t = make_tensor_3d((BATCH, N_BITS, 6), x)
    out_t = make_tensor_3d((BATCH, N_BITS, 2))
    g_out = make_tensor_3d((BATCH, N_BITS, 2), np.ones_like(
        np.zeros((BATCH, N_BITS, 2), dtype=np.float32)))
    g_in = make_tensor_3d((BATCH, N_BITS, 6))

    expert = RecurrentExpert(hidden_size=HIDDEN, name="demo_expert")

    expert.forward(x_t, out_t)
    out = np.ctypeslib.as_array(
        out_t.contents.data[: BATCH * N_BITS * 2]
    ).reshape(BATCH, N_BITS, 2)
    assert np.all(np.isfinite(out)), "forward produced non-finite values"

    result_prob = 1.0 / (1.0 + np.exp(-out[:, :, 0]))
    matches = int((result_prob.round() == result_bits).sum())
    print(f"output shape: {out.shape} (batch, timesteps, [result, carry])")
    print(f"sample predictions vs ground-truth result bits (untrained):")
    for i in range(min(4, BATCH)):
        pred = "".join(str(int(v)) for v in result_prob[i])
        truth = "".join(str(int(v)) for v in result_bits[i])
        print(f"  {a[i]:2d} + {b[i]:2d}: pred={pred}  true={truth}")
    print(f"bit matches: {matches}/{BATCH * N_BITS}")

    # Backward must produce finite gradients for the inputs.
    expert.backward(out_t, g_in)
    g = np.ctypeslib.as_array(g_in.contents.data[: BATCH * N_BITS * 6])
    assert np.all(np.isfinite(g)), "backward produced non-finite gradients"
    print(f"backward ok — grad_input shape {g_in.contents.shape[:3]}")

    expert.close()


if __name__ == "__main__":
    main()
