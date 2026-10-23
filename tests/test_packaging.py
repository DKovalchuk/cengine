"""
Packaging tests: how the ctypes layer locates libcengine.so.

The library must be discoverable in three layouts:

  * installed    — `pip install .` ships libcengine.so inside the cengine
                   package, next to bindings.py;
  * development  — a repo checkout keeps the `make` product under
                   build/lib/;
  * explicit     — CENGINE_LIB environment variable overrides everything.

Run: pytest tests/test_packaging.py -v
"""
import os
import sys

import pytest

# ── Path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cengine import bindings

PACKAGE_DIR = os.path.dirname(os.path.abspath(bindings.__file__))


def test_candidates_start_with_packaged_library(monkeypatch):
    """The package-adjacent libcengine.so is the primary candidate."""
    monkeypatch.delenv("CENGINE_LIB", raising=False)
    cands = bindings._library_candidates()
    assert cands[0] == os.path.join(PACKAGE_DIR, "libcengine.so")
    # System fallback stays last: resolved through LD_LIBRARY_PATH.
    assert cands[-1] == "libcengine.so"


def test_candidates_include_repo_build_dir(monkeypatch):
    """A development checkout keeps the make product under build/lib/."""
    monkeypatch.delenv("CENGINE_LIB", raising=False)
    cands = [os.path.normpath(c) for c in bindings._library_candidates()]
    repo_build = os.path.normpath(
        os.path.join(PACKAGE_DIR, "..", "build", "lib", "libcengine.so")
    )
    assert repo_build in cands


def test_env_override_wins(monkeypatch, tmp_path):
    """CENGINE_LIB overrides every other candidate."""
    fake = tmp_path / "custom_libcengine.so"
    monkeypatch.setenv("CENGINE_LIB", str(fake))
    cands = bindings._library_candidates()
    assert cands[0] == str(fake)


def test_library_loads():
    """The discovered library actually loads in this checkout."""
    assert bindings.lib is not None
    assert bindings.lib.ce_init  # symbol present


def test_pyproject_exists():
    """`pip install .` requires packaging metadata at the repo root."""
    assert os.path.isfile(os.path.join(PROJECT_ROOT, "pyproject.toml")), (
        "pyproject.toml missing — the package cannot be installed with pip"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
