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
    if _STATE.plotter is None and fallback is not None:
        _STATE.plotter = fallback
    return _STATE.plotter


def set_plotter(value: Any) -> None:
    _STATE.plotter = value


def get_notebook_plotter(fallback=None):
    if _STATE.notebook_plotter is None and fallback is not None:
        _STATE.notebook_plotter = fallback
    return _STATE.notebook_plotter


def set_notebook_plotter(value: Any) -> None:
    _STATE.notebook_plotter = value


def get_notebook_backend(fallback=None):
    if _STATE.notebook_backend is None and fallback is not None:
        _STATE.notebook_backend = fallback
    return _STATE.notebook_backend


def set_notebook_backend(value: Any) -> None:
    _STATE.notebook_backend = value


def get_last_figure(fallback=None):
    if _STATE.last_figure is None and fallback is not None:
        _STATE.last_figure = fallback
    return _STATE.last_figure


def set_last_figure(value: Any) -> None:
    _STATE.last_figure = value
