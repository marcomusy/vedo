#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Compatibility facade for vedo core algorithm mixins."""

from vedo.lazy_imports import build_attr_map, dir_lazy, getattr_lazy

_LAZY_EXPORT_MAP, __all__ = build_attr_map(
    ("vedo.core.data", ["DataArrayHelper"]),
    ("vedo.core.common", ["CommonAlgorithms"]),
    ("vedo.core.points", ["PointAlgorithms"]),
    ("vedo.core.volume", ["VolumeAlgorithms"]),
)


def __getattr__(name):
    return getattr_lazy(__name__, globals(), name, attr_map=_LAZY_EXPORT_MAP)


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP)
