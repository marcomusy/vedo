#!/usr/bin/env python3
from __future__ import annotations

"""Lightweight console entry point for the ``vedo`` command."""

import os
import runpy

_CLI_MODULE = runpy.run_path(os.path.join(os.path.dirname(__file__), "vedo", "cli.py"))

main = _CLI_MODULE["main"]

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
