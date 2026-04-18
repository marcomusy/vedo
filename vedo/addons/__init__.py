#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Compatibility facade for add-on actors, widgets and utilities."""

from vedo.lazy_imports import build_attr_map, dir_lazy, getattr_lazy

__docformat__ = "google"

__doc__ = """
Create additional objects like axes, legends, lights, etc.

![](https://vedo.embl.es/images/pyplot/customAxes2.png)
"""

_LAZY_EXPORT_MAP, __all__ = build_attr_map(
    ("vedo.addons.core", ["Goniometer", "Light", "ScalarBar", "ScalarBar3D"]),
    ("vedo.addons.axes", ["Axes", "add_global_axes"]),
    ("vedo.addons.widgets", ["ButtonWidget", "Button", "DrawingWidget"]),
    ("vedo.addons.interaction", ["LineWidget", "PointCloudWidget", "SplineTool"]),
    ("vedo.addons.ui", ["Flagpost", "Icon", "LegendBox"]),
    ("vedo.addons.sliders", ["SliderWidget", "Slider2D", "Slider3D"]),
    (
        "vedo.addons.cutters",
        [
            "BaseCutter",
            "PlaneCutter",
            "BoxCutter",
            "SphereCutter",
            "RendererFrame",
            "ProgressBarWidget",
        ],
    ),
    (
        "vedo.addons.measure",
        ["Ruler2D", "Ruler3D", "RulerAxes", "DistanceTool", "compute_visible_bounds"],
    ),
)


def __getattr__(name):
    return getattr_lazy(__name__, globals(), name, attr_map=_LAZY_EXPORT_MAP)


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP)
