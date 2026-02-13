#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for add-on actors, widgets and utilities."""

from vedo.addons_core import Goniometer, Light, ScalarBar, ScalarBar3D
from vedo.addons_axes import Axes, add_global_axes
from vedo.addons_widgets import ButtonWidget, Button, DrawingWidget
from vedo.addons_interaction import PointCloudWidget, SplineTool
from vedo.addons_ui import Flagpost, LegendBox
from vedo.addons_sliders import SliderWidget, Slider2D, Slider3D
from vedo.addons_cutters import (
    BaseCutter,
    PlaneCutter,
    BoxCutter,
    SphereCutter,
    RendererFrame,
    ProgressBarWidget,
)
from vedo.addons_icon import Icon
from vedo.addons_measure import (
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
