"""
Regression tests for the flat input/target layout (Python-layer finding P1).

_batch_to_tensors must anchor rows on the MODEL's dimensions — exactly like
ce_trainer_prepare_batch in C:

    final    = (max_input_dim - 3) // 2
    input  = [a_bits(0..n), b_bits(final..final+n), op_code(2*final..+3)]
    target = [result(0..n), carry(t_anchor..t_anchor+n)]  t_anchor = max_target_dim // 2

The old implementation anchored b/op/carry on the BATCH's max n_bits, so
whenever a batch was narrower than the model (batch n_bits < model bit
width) the data landed in columns the model reads as different features:
op_code sat in the b_bits columns, carry sat in result columns, and the
mask marked the wrong positions (loss compared result slots against carry
slots).

Run: pytest tests/test_batch_layout.py -v
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cengine.bindings import CE_OK, c_int, c_int64, check, lib
from cengine.trainer import _batch_to_tensors

check(lib.ce_init())

# Model for max_bits=8: input_dim = 19 (final = 8), target_dim = 16
MODEL_IN = 19
MODEL_TGT = 16
IN_ANCHOR = (MODEL_IN - 3) // 2      # 8
TGT_ANCHOR = MODEL_TGT // 2          # 8


def _make_batch(min_bits: int, max_bits: int, ops: list[int], seed: int = 7):
    """Return (batch_ptr, dataloader) — caller destroys both."""
    ops_arr = (c_int * len(ops))(*ops)
    gen = lib.ce_alu_gen_create(
        c_int(min_bits), c_int(max_bits), ops_arr, c_int(len(ops)),
        c_int64(seed),
    )
    assert gen, "ce_alu_gen_create returned NULL"
    dl = lib.ce_dataloader_create(gen, c_int(4), c_int(16), c_int64(seed))
    assert dl, "ce_dataloader_create returned NULL"
    batch = lib.ce_dataloader_next(dl)
    assert batch, "ce_dataloader_next returned NULL"
    return batch, dl, gen


def _sample_fields(sample_ptr, sn: int):
    s = sample_ptr.contents
    a = [s.a_bits.contents.data[k] for k in range(sn)]
    b = [s.b_bits.contents.data[k] for k in range(sn)]
    op = [s.op_code.contents.data[k] for k in range(3)]
    res = [s.result_bits.contents.data[k] for k in range(sn)]
    carry = [s.carry_bits.contents.data[k] for k in range(sn)]
    return a, b, op, res, carry, s.n_bits


def test_narrow_batch_is_model_anchored():
    """n_bits(batch)=4 under a width-19 model → rows follow final=8."""
    # Force a non-zero op encoding (AND = 2 → [0,1,0]) so the op-anchor
    # assertion cannot pass vacuously on an all-zero opcode.
    batch, dl, _gen = _make_batch(4, 4, ops=[2], seed=7)
    try:
        inp, tgt, n_bits, msk = _batch_to_tensors(
            batch, max_input_dim=MODEL_IN, max_target_dim=MODEL_TGT,
        )
        assert n_bits == 4
        assert (inp.contents.shape[0], inp.contents.shape[1]) == (4, MODEL_IN)
        assert (tgt.contents.shape[0], tgt.contents.shape[1]) == (4, MODEL_TGT)

        batch_c = batch.contents
        for i in range(batch_c.count):
            a, b, op, res, carry, sn = _sample_fields(batch_c.samples[i], sn=4)
            row = [inp.contents.data[i * MODEL_IN + j] for j in range(MODEL_IN)]
            trow = [tgt.contents.data[i * MODEL_TGT + j] for j in range(MODEL_TGT)]
            mrow = [msk.contents.data[i * MODEL_TGT + j] for j in range(MODEL_TGT)]

            # input anchors (mirror C prepare_batch: final = 8)
            assert row[0:sn] == a, f"row{i}: a_bits not at 0"
            assert row[IN_ANCHOR:IN_ANCHOR + sn] == b, (
                f"row{i}: b_bits not at model anchor {IN_ANCHOR} "
                f"(batch-n_bits anchoring leaks b_bits into model columns)"
            )
            assert row[2 * IN_ANCHOR:2 * IN_ANCHOR + 3] == op, (
                f"row{i}: op_code not at model anchor {2 * IN_ANCHOR} "
                f"(op landed in the model's b_bits columns instead)"
            )
            # region between a-end and model b-anchor must be padding
            assert row[sn:IN_ANCHOR] == [0.0] * (IN_ANCHOR - sn), (
                f"row{i}: model b-region contains non-padding data"
            )

            # target anchors
            assert trow[0:sn] == res, f"row{i}: result not at 0"
            assert trow[TGT_ANCHOR:TGT_ANCHOR + sn] == carry, (
                f"row{i}: carry not at model anchor {TGT_ANCHOR} "
                f"(loss would compare model result slots against carry)"
            )

            # mask follows the model anchors too
            assert mrow[0:sn] == [1.0] * sn, f"row{i}: result mask"
            assert mrow[sn:TGT_ANCHOR] == [0.0] * (TGT_ANCHOR - sn), (
                f"row{i}: model result-padding marked valid"
            )
            assert mrow[TGT_ANCHOR:TGT_ANCHOR + sn] == [1.0] * sn, (
                f"row{i}: carry region not marked valid"
            )
            assert mrow[TGT_ANCHOR + sn:] == [0.0] * (MODEL_TGT - TGT_ANCHOR - sn), (
                f"row{i}: carry-padding marked valid"
            )
    finally:
        lib.ce_batch_destroy(batch)
        lib.ce_dataloader_destroy(dl)


def test_no_model_dims_keeps_batch_anchor():
    """Without model dims the legacy batch-n_bits anchor must remain."""
    batch, dl, _gen = _make_batch(2, 6, ops=[0, 1, 2, 3], seed=11)
    try:
        inp, tgt, n_bits, msk = _batch_to_tensors(batch)   # max dims = 0
        width = inp.contents.shape[1]
        assert width == 2 * n_bits + 3
        assert tgt.contents.shape[1] == 2 * n_bits

        s = batch.contents.samples[0].contents
        sn = s.n_bits
        a, b, op, res, carry, _ = _sample_fields(batch.contents.samples[0], sn)
        row = [inp.contents.data[j] for j in range(width)]
        assert row[0:sn] == a
        assert row[n_bits:n_bits + sn] == b
        assert row[2 * n_bits:2 * n_bits + 3] == op
        trow = [tgt.contents.data[j] for j in range(2 * n_bits)]
        assert trow[0:sn] == res
        assert trow[n_bits:n_bits + sn] == carry
    finally:
        lib.ce_batch_destroy(batch)
        lib.ce_dataloader_destroy(dl)


def test_full_width_batch_matches_model_anchor():
    """n_bits == model width → both anchors coincide (pins equality)."""
    batch, dl, _gen = _make_batch(8, 8, ops=[1, 3], seed=13)
    try:
        inp, tgt, n_bits, msk = _batch_to_tensors(
            batch, max_input_dim=MODEL_IN, max_target_dim=MODEL_TGT,
        )
        assert n_bits == 8
        assert inp.contents.shape[1] == MODEL_IN
        batch_c = batch.contents
        for i in range(batch_c.count):
            a, b, op, res, carry, sn = _sample_fields(batch_c.samples[i], sn=8)
            row = [inp.contents.data[i * MODEL_IN + j] for j in range(MODEL_IN)]
            trow = [tgt.contents.data[i * MODEL_TGT + j] for j in range(MODEL_TGT)]
            assert row[0:sn] == a
            assert row[IN_ANCHOR:IN_ANCHOR + sn] == b
            assert row[2 * IN_ANCHOR:2 * IN_ANCHOR + 3] == op
            assert trow[0:sn] == res
            assert trow[TGT_ANCHOR:TGT_ANCHOR + sn] == carry
    finally:
        lib.ce_batch_destroy(batch)
        lib.ce_dataloader_destroy(dl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
