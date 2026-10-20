"""
Regression tests for the trainer flat batch layout.

op_code is a 3-float vector (3-bit binary encoding, data.h: [3] —
CE_OP_OR=4 needs bit 2), so a flat input row must reserve 3 op slots:
dim = 2*n_bits + 3.  Writing 3 floats into a 2-slot row pushed the third
float into the next row / one past the input buffer (heap OOB write in
trainer.c prepare_batch) and the model never saw op bit 2.

Run: pytest tests/test_trainer_layout.py -v
"""
import os
import sys

import pytest

# ── Path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from python.bindings import lib, c_int, c_int64, check
from python.trainer import _batch_to_tensors

# Ensure C engine is initialized
check(lib.ce_init())


@pytest.fixture()
def batch_ptr():
    """Build a small ALU batch (4 samples, 4..8 bits, ops 0..3)."""
    ops = (c_int * 4)(0, 1, 2, 3)
    gen = lib.ce_alu_gen_create(c_int(4), c_int(8), ops, c_int(4), c_int64(42))
    assert gen, "ce_alu_gen_create returned NULL"

    dl = lib.ce_dataloader_create(gen, c_int(4), c_int(16), c_int64(42))
    assert dl, "ce_dataloader_create returned NULL"

    batch = lib.ce_dataloader_next(dl)
    assert batch, "ce_dataloader_next returned NULL"
    yield batch

    lib.ce_dataloader_destroy(dl)
    # ce_alu_gen has no destroy binding; freed at process exit.


def test_flat_layout_reserves_three_op_slots(batch_ptr):
    """Flat input row width must be 2*n_bits+3, not 2*n_bits+2.

    op_code copies 3 floats at offset 2*n_bits — with width 2n+2 the
    third float lands outside the row (heap OOB write on the last row).
    """
    input_t, target_t, n_bits, mask_t = _batch_to_tensors(
        batch_ptr, max_input_dim=0, max_target_dim=0, recurrent=False
    )

    width = input_t.contents.shape[1]
    assert width == 2 * n_bits + 3, (
        f"flat input width {width} != 2*{n_bits}+3 — op_code (3 floats) "
        f"spills out of the row: heap OOB write"
    )

    # The op_code region must sit strictly inside every row.
    op_start = 2 * n_bits
    assert op_start + 3 <= width, "op_code region crosses the row boundary"

    lib.ce_tensor_release(input_t)
    lib.ce_tensor_release(target_t)
    lib.ce_tensor_release(mask_t)


def test_flat_layout_op_bits_are_copied(batch_ptr):
    """All three op bits from the sample must land inside row 0."""
    input_t, target_t, n_bits, mask_t = _batch_to_tensors(
        batch_ptr, max_input_dim=0, max_target_dim=0, recurrent=False
    )

    batch = batch_ptr.contents
    sample0 = batch.samples[0].contents
    width = input_t.contents.shape[1]
    op_start = 2 * n_bits

    for bit in range(3):
        expected = sample0.op_code.contents.data[bit]
        got = input_t.contents.data[op_start + bit]
        assert got == pytest.approx(expected, abs=1e-6), (
            f"op bit {bit}: got {got}, expected {expected} "
            f"— op_code does not fit in the row"
        )

    lib.ce_tensor_release(input_t)
    lib.ce_tensor_release(target_t)
    lib.ce_tensor_release(mask_t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
