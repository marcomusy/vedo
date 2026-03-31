#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Ready-to-use interactive application plotters and tools."""

from vedo.lazy_imports import build_attr_map, dir_lazy, getattr_lazy

__docformat__ = "google"

__doc__ = """
This module contains vedo applications which provide some *ready-to-use* funcionalities

<img src="https://vedo.embl.es/images/advanced/app_raycaster.gif" width="500">
"""

_LAZY_EXPORT_MAP, __all__ = build_attr_map(
    (
        "vedo.applications.slicing",
        ["Slicer3DPlotter", "Slicer3DTwinPlotter", "Slicer2DPlotter", "RayCastPlotter"],
    ),
    (
        "vedo.applications.morphing",
        ["MorphPlotter", "MorphByLandmarkPlotter", "MorphBySplinesPlotter"],
    ),
    ("vedo.applications.browsers", ["IsosurfaceBrowser", "Browser"]),
    ("vedo.applications.editing", ["FreeHandCutPlotter", "SplinePlotter", "ImageEditor"]),
    ("vedo.applications.animation", ["Animation", "AnimationPlayer", "Clock"]),
)


def __getattr__(name):
    return getattr_lazy(__name__, globals(), name, attr_map=_LAZY_EXPORT_MAP)


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP)
