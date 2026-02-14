#!/usr/bin/env python3
from __future__ import annotations
"""Helper utilities for optional dependencies in examples."""

import importlib
import sys


def require_module(module_name: str, pip_name: str | None = None):
    """Import optional module, print a skip message, and exit cleanly if missing."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pkg = pip_name or module_name
        print(f"Skipping example: optional dependency '{module_name}' is not installed.")
        print(f"Install with: pip install {pkg}")
        sys.exit(0)
