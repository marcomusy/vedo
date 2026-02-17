#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Compatibility facade for add-on actors, widgets and utilities."""

from vedo.addons.core import Goniometer, Light, ScalarBar, ScalarBar3D
from vedo.addons.axes import Axes, add_global_axes
from vedo.addons.widgets import ButtonWidget, Button, DrawingWidget
from vedo.addons.interaction import PointCloudWidget, SplineTool
from vedo.addons.ui import Flagpost, LegendBox
from vedo.addons.sliders import SliderWidget, Slider2D, Slider3D
from vedo.addons.cutters import (
    BaseCutter,
    PlaneCutter,
    BoxCutter,
    SphereCutter,
    RendererFrame,
    ProgressBarWidget,
)
from vedo.addons.icon import Icon
from vedo.addons.measure import (
    Ruler2D,
    Ruler3D,
    RulerAxes,
    DistanceTool,
    compute_visible_bounds,
)

__docformat__ = "google"

__doc__ = """
Create additional objects like axes, legends, lights, etc.

![](https://vedo.embl.es/images/pyplot/customAxes2.png)
"""

__all__ = [
    "ScalarBar",
    "ScalarBar3D",
    "Slider2D",
    "Slider3D",
    "Icon",
    "LegendBox",
    "Light",
    "Axes",
    "RendererFrame",
    "Ruler2D",
    "Ruler3D",
    "RulerAxes",
    "DistanceTool",
    "SplineTool",
    "DrawingWidget",
    "Goniometer",
    "Button",
    "ButtonWidget",
    "Flagpost",
    "ProgressBarWidget",
    "BoxCutter",
    "PlaneCutter",
    "SphereCutter",
]
