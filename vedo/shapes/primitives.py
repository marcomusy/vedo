#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Primitive geometric shapes.

Compatibility facade that re-exports primitive symbols from focused modules.
"""

from vedo.shapes.primitives_planar import (
    Triangle,
    Polygon,
    Circle,
    GeoCircle,
    Star,
    Disc,
    IcoSphere,
    Sphere,
    Spheres,
    Earth,
    Ellipsoid,
    Grid,
    Plane,
    Rectangle,
)
from vedo.shapes.primitives_solids import (
    Box,
    Cube,
    TessellatedBox,
    Spring,
    Cylinder,
    Cone,
    Pyramid,
    Torus,
    Paraboloid,
    Hyperboloid,
)

__all__ = [
    "Triangle",
    "Polygon",
    "Circle",
    "GeoCircle",
    "Star",
    "Disc",
    "IcoSphere",
    "Sphere",
    "Spheres",
    "Earth",
    "Ellipsoid",
    "Grid",
    "Plane",
    "Rectangle",
    "Box",
    "Cube",
    "TessellatedBox",
    "Spring",
    "Cylinder",
    "Cone",
    "Pyramid",
    "Torus",
    "Paraboloid",
    "Hyperboloid",
]
