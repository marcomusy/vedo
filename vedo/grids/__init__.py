#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Grid datasets package facade."""

from vedo.lazy_imports import build_attr_map, dir_lazy, getattr_lazy

_LAZY_EXPORT_MAP, __all__ = build_attr_map(
    ("vedo.grids.unstructured", ["UnstructuredGrid"]),
    ("vedo.grids.tetmesh", ["TetMesh"]),
    ("vedo.grids.rectilinear", ["RectilinearGrid"]),
    ("vedo.grids.structured", ["StructuredGrid"]),
    ("vedo.grids.explicit", ["ExplicitStructuredGrid"]),
)


def __getattr__(name):
    return getattr_lazy(__name__, globals(), name, attr_map=_LAZY_EXPORT_MAP)


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP)
