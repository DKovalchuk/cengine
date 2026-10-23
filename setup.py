"""
Build hook: compile the C engine before packaging the python files.

`pip install .` runs `make shared` and ships the resulting libcengine.so
inside the cengine package, so an installed package is self-contained —
ctypes finds the library next to bindings.py (see `_library_candidates`).

Metadata lives in pyproject.toml; this file only carries the native build.
"""
import os
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = os.path.dirname(os.path.abspath(__file__))


class BuildPyWithNativeLibrary(build_py):
    def run(self):
        super().run()
        subprocess.run(["make", "shared"], cwd=ROOT, check=True)
        built = os.path.join(ROOT, "build", "lib", "libcengine.so")
        target = os.path.join(self.build_lib, "cengine")
        os.makedirs(target, exist_ok=True)
        shutil.copy2(built, os.path.join(target, "libcengine.so"))


setup(cmdclass={"build_py": BuildPyWithNativeLibrary})
