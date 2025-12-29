#!/usr/bin/env python3
"""
Stable entrypoint wrapper for the main pipeline script.

This wrapper keeps the public command stable while allowing you to iterate on
the underlying implementation (currently predict_science_figs_v6_4.py).

Usage:
  python3 scripts/predict_science_figs.py --help
"""
from __future__ import annotations
import runpy
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TARGET = HERE / "predict_science_figs_v6_4.py"

if __name__ == "__main__":
    # Ensure the target runs as __main__ and receives the same argv.
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
