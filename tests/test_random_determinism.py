"""
Pin: python tests must not touch the process-wide numpy RNG.

Every test that draws random data (or reseeds with np.random.seed) reads
or corrupts the global MT19937 state, which makes test outcomes depend on:

  * test execution order (-k filters, deselection, reordering),
  * OS entropy — the first draw in a process starts from random entropy,
    so the same suite can produce different inputs on every CI run.

Either dependency is a flake generator. The correct pattern is a test-local
generator (np.random.default_rng(seed)) that neither reads nor writes the
global state.

This module executes every test function of test_recurrent_cengine between
two snapshots of np.random.get_state() and asserts the state is untouched.

Usage:
    pytest tests/test_random_determinism.py -v
"""
import os
import sys

import numpy as np
import pytest

# ── Path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_recurrent_cengine as rec


def _all_test_functions():
    """Discover every pytest-style test function in the recurrent module."""
    found = []
    for cls_name in sorted(vars(rec)):
        cls = getattr(rec, cls_name)
        if not (isinstance(cls, type) and cls_name.startswith("Test")):
            continue
        for fn_name in sorted(vars(cls)):
            if fn_name.startswith("test_") and callable(getattr(cls, fn_name)):
                found.append((cls_name, fn_name))
    return found


@pytest.mark.parametrize(
    "cls_name,fn_name",
    _all_test_functions(),
    ids=lambda v: str(v),
)
def test_does_not_touch_global_rng_state(cls_name, fn_name):
    """A test that draws from (or reseeds) the global RNG is order-dependent.

    Run the test function between two snapshots of the process-wide
    numpy RNG state: any draw or np.random.seed call changes it, which
    means the test's inputs depend on what ran before it and on the
    entropy the process started with.
    """
    np.random.seed(20261023)
    before = np.random.get_state()

    cls = getattr(rec, cls_name)
    getattr(cls(), fn_name)()

    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2] == after[2], (
        f"{cls_name}.{fn_name} changed the global numpy RNG state "
        f"(position {before[2]} -> {after[2]}): the test draws from or "
        f"reseeds np.random, so its inputs depend on test order and OS "
        f"entropy. Use a test-local np.random.default_rng(seed) instead."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
