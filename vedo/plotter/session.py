#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Centralized runtime session state for vedo.

This module intentionally has no imports from ``vedo`` to avoid import cycles.
"""

from typing import Any


class _SessionState:
    __slots__ = ("plotter", "notebook_plotter", "notebook_backend", "last_figure")

    def __init__(self) -> None:
        self.plotter = None
        self.notebook_plotter = None
        self.notebook_backend = None
        self.last_figure = None


_STATE = _SessionState()


def get_plotter(fallback=None):
    return _STATE.plotter if _STATE.plotter is not None else fallback


def set_plotter(value: Any) -> None:
    _STATE.plotter = value


def get_notebook_plotter(fallback=None):
    return _STATE.notebook_plotter if _STATE.notebook_plotter is not None else fallback


def set_notebook_plotter(value: Any) -> None:
    _STATE.notebook_plotter = value


def get_notebook_backend(fallback=None):
    return _STATE.notebook_backend if _STATE.notebook_backend is not None else fallback


def set_notebook_backend(value: Any) -> None:
    _STATE.notebook_backend = value


def get_last_figure(fallback=None):
    return _STATE.last_figure if _STATE.last_figure is not None else fallback


def set_last_figure(value: Any) -> None:
    _STATE.last_figure = value
