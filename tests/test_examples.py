"""
Smoke tests: every example must run to completion.

Examples are the first thing a new user copies — a stale or broken one is
a bug. Each script under examples/ is executed as a subprocess with a
timeout; exit code must be 0.

Run: pytest tests/test_examples.py -v
"""
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")

EXAMPLES = [
    "01_quickstart_alu.py",
    "02_checkpoints_telemetry.py",
    "03_manual_training_loop.py",
    "04_recurrent_expert.py",
    "05_pytorch_cuda.py",
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_runs(example):
    """The example executes end-to-end with exit code 0."""
    path = os.path.join(EXAMPLES_DIR, example)
    assert os.path.isfile(path), f"missing example: examples/{example}"

    proc = subprocess.run(
        [sys.executable, path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"examples/{example} exited with {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
