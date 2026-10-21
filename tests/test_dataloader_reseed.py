"""
Regression tests for the Python-side generator reconfiguration (major A1).

The trainer used to mutate the shared generator directly
(``rng_state``/``n_ops``/``ops``) while the C dataloader's prefetch worker
thread was reading it in ``ce_alu_gen_batch()`` — a data race.  The fix
routes every epoch-boundary reconfiguration through
``ce_dataloader_reseed()``, which joins the worker first.

Two guards here:
1. bindings must declare the full 64-bit signature of the reseed call
   (without argtypes ctypes truncates seeds to 32 bits / passes pointers
   wrong), verified functionally against a fresh generator;
2. the trainer source must not mutate generator fields directly — that is
   exactly the racy pattern this fix removes.

Run: pytest tests/test_dataloader_reseed.py -v
"""
import os

import pytest

# ── Path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path_added = False
import sys

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys_path_added = True

from python.bindings import CE_OK, c_int, c_int64, check, lib

check(lib.ce_init())


def _make_batch_equal(a, b):
    if a is None or b is None:
        return False
    if a.contents.count != b.contents.count:
        return False
    for i in range(a.contents.count):
        sa = a.contents.samples[i].contents
        sb = b.contents.samples[i].contents
        if sa.n_bits != sb.n_bits or sa.op != sb.op:
            return False
        for field in ("a_bits", "b_bits", "result_bits"):
            ta = getattr(sa, field).contents
            tb = getattr(sb, field).contents
            if ta.numel != tb.numel:
                return False
            for j in range(ta.numel):
                if ta.data[j] != tb.data[j]:
                    return False
    return True


@pytest.fixture()
def dataloader():
    ops = (c_int * 4)(0, 1, 2, 3)
    gen = lib.ce_alu_gen_create(c_int(4), c_int(8), ops, c_int(4), c_int64(7))
    assert gen, "ce_alu_gen_create returned NULL"
    dl = lib.ce_dataloader_create(gen, c_int(4), c_int(16), c_int64(7))
    assert dl, "ce_dataloader_create returned NULL"
    yield dl
    lib.ce_dataloader_destroy(dl)
    # ce_alu_gen has no destroy binding; freed at process exit.


def test_reseed_binding_passes_full_64bit_seed(dataloader):
    """reseeding with a seed above 2^31 must survive the ctypes boundary.

    Without ``argtypes`` declaring ``c_int64``, ctypes truncates the Python
    int to 32 bits and the dataloader stream diverges from a fresh
    generator created with the same seed.
    """
    seed = (1 << 40) + 7

    ops = (c_int * 4)(0, 1, 2, 3)
    ref_gen = lib.ce_alu_gen_create(c_int(4), c_int(8), ops, c_int(4), c_int64(seed))
    assert ref_gen, "ref gen create"
    ref = lib.ce_alu_gen_batch(ref_gen, 4)
    assert ref, "ref batch"

    status = lib.ce_dataloader_reseed(dataloader, seed, None, 0)
    assert status == CE_OK, f"ce_dataloader_reseed returned {status}"

    got = lib.ce_dataloader_next(dataloader)
    assert got, "dataloader_next returned NULL"
    assert _make_batch_equal(got, ref), (
        "batch after ce_dataloader_reseed differs from a fresh generator "
        "with the same seed — seed was likely truncated at the ctypes "
        "boundary (missing c_int64 argtype)"
    )


def test_reseed_binding_rejects_bad_op_filter(dataloader):
    """Invalid op filters must be rejected through the bindings as well."""
    bad_ops = (c_int * 1)(5)  # CE_NUM_OPS == 5 → out of range
    status = lib.ce_dataloader_reseed(dataloader, 42, bad_ops, 1)
    assert status != CE_OK, "out-of-range op accepted by ce_dataloader_reseed"


def test_trainer_does_not_mutate_generator_fields():
    """The trainer must not poke generator internals directly.

    Direct writes to ``rng_state``/``n_ops``/``ops`` race with the
    prefetch worker; all reconfiguration goes through
    ``ce_dataloader_reseed()`` which joins the worker first.
    """
    trainer_path = os.path.join(PROJECT_ROOT, "python", "trainer.py")
    with open(trainer_path, encoding="utf-8") as f:
        src = f.read()

    assert ".rng_state =" not in src, (
        "python/trainer.py writes rng_state directly — races with the "
        "dataloader prefetch worker; use ce_dataloader_reseed()"
    )
    assert "gen.n_ops =" not in src, (
        "python/trainer.py writes n_ops directly — races with the "
        "dataloader prefetch worker; use ce_dataloader_reseed()"
    )
    assert "gen.ops[" not in src, (
        "python/trainer.py writes ops directly — races with the "
        "dataloader prefetch worker; use ce_dataloader_reseed()"
    )
    assert "ce_dataloader_reseed" in src, (
        "python/trainer.py does not use ce_dataloader_reseed"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
