"""
Regression tests for module lifecycle (Python-layer finding P6).

Nothing ever destroyed module handles: Module.__del__ was `pass` and
Sequential had no C destroy at all — every Python-built model leaked its
C-side container, children and parameter/gradient lists for the whole
process (the docstring claim "children are destroyed when the container
is destroyed" had no implementation behind it).

Now:
  * C gained ce_sequential_destroy() — dispatch-destroys children
    recursively (nested containers own THEIR children too), frees the
    container state, then the generic module destroy;
  * Python gained Module.close() (idempotent) with __del__ wiring, and
    Sequential.close() which destroys the whole C subtree and then
    invalidates every Python handle pointing into it;
  * a leaf owned by a container never self-destroys (the container is
    the single owner) — no double-free window.

Run: pytest tests/test_module_lifecycle.py -v
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python.bindings import CEngineError, lib
from python.module import Linear, Sequential

check_init = getattr(lib, "ce_init", None)
if check_init:
    check_init()


def test_sequential_destroy_binding_exists():
    assert hasattr(lib, "ce_sequential_destroy"), (
        "ce_sequential_destroy binding missing — containers can never "
        "be freed from Python"
    )


def test_leaf_close_is_idempotent():
    lin = Linear(4, 4, name="leaf_close")
    assert lin._handle is not None

    lin.close()
    assert lin._handle is None
    with pytest.raises(CEngineError):
        _ = lin.handle

    lin.close()          # idempotent — must not double-free
    assert lin._handle is None


def test_container_close_invalidates_whole_subtree():
    inner = Sequential(
        Linear(16, 8, name="in0"),
        Linear(8, 4, name="in1"),
        name="inner",
    )
    model = Sequential(
        Linear(19, 16, name="top0"),
        inner,
        Linear(16, 4, name="top1"),
        name="outer",
    )

    model.close()                      # RED: AttributeError (no close)

    assert model._handle is None
    with pytest.raises(CEngineError):
        _ = model.handle

    # Every handle into the destroyed C subtree must be invalidated —
    # touching them after the close would be use-after-free.
    assert inner._handle is None, "nested container handle still live"
    for child in model:
        assert child._handle is None, "direct child handle still live"
    for child in inner:
        assert child._handle is None, "grandchild handle still live"

    model.close()                      # idempotent
    inner.close()                      # closing an invalidated subtree is safe


def test_leaf_inside_container_never_self_destroys():
    """A container owns its children: an explicit leaf close is a no-op,
    so GC'ing a child reference can never free memory the container uses."""
    leaf = Linear(4, 4, name="owned_leaf")
    container = Sequential(Linear(19, 16, name="other"), leaf, name="owning")

    leaf.close()                       # owned by container → no-op
    assert leaf._handle is not None, (
        "leaf self-destroyed while the container still owns it"
    )

    container.close()
    assert leaf._handle is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
