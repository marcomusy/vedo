#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Curve, line and arrow shapes.

Compatibility facade that re-exports curve symbols from focused modules.
"""

from vedo.shapes.curves_core import (
    Line,
    DashedLine,
    RoundedLine,
    Lines,
    Arc,
    Spline,
    KSpline,
    CSpline,
    Bezier,
    NormalLines,
    Tube,
    ThickTube,
    Tubes,
)
from vedo.shapes.curves_extras import Ribbon, Arrow, Arrows, Arrow2D, Arrows2D, FlatArrow

__all__ = [
    "Line",
    "DashedLine",
    "RoundedLine",
    "Lines",
    "Arc",
    "Spline",
    "KSpline",
    "CSpline",
    "Bezier",
    "NormalLines",
    "Tube",
    "ThickTube",
    "Tubes",
    "Ribbon",
    "Arrow",
    "Arrows",
    "Arrow2D",
    "Arrows2D",
    "FlatArrow",
]
