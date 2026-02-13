#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ready-to-use interactive application plotters and tools."""

from vedo.applications.slicing import (
    Slicer3DPlotter,
    Slicer3DTwinPlotter,
    Slicer2DPlotter,
    RayCastPlotter,
)
from vedo.applications.morphing import (
    MorphPlotter,
    MorphByLandmarkPlotter,
    MorphBySplinesPlotter,
)
from vedo.applications.browsers import IsosurfaceBrowser, Browser
from vedo.applications.editing import FreeHandCutPlotter, SplinePlotter, ImageEditor
from vedo.applications.animation import Animation, AnimationPlayer, Clock

__docformat__ = "google"

__doc__ = """
This module contains vedo applications which provide some *ready-to-use* funcionalities

<img src="https://vedo.embl.es/images/advanced/app_raycaster.gif" width="500">
"""

__all__ = [
    "Browser",
    "IsosurfaceBrowser",
    "FreeHandCutPlotter",
    "RayCastPlotter",
    "Slicer2DPlotter",
    "Slicer3DPlotter",
    "Slicer3DTwinPlotter",
    "MorphPlotter",
    "MorphByLandmarkPlotter",
    "MorphBySplinesPlotter",
    "SplinePlotter",
    "ImageEditor",
    "Animation",
    "AnimationPlayer",
    "Clock",
]
