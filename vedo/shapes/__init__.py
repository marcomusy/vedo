#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Submodule to generate simple and complex geometric shapes."""

from vedo._lazy import build_attr_map, dir_lazy, getattr_lazy


def VedoLogo(distance=0.0, c=None, bc="t", version=False, frame=True):
    """Create the 3D vedo logo."""
    from vedo.shapes.branding import vedo_logo

    return vedo_logo(distance=distance, c=c, bc=bc, version=version, frame=frame)


_LAZY_EXPORT_MAP, _LAZY_EXPORTS = build_attr_map(
    ("vedo.shapes.glyphs", ["Glyph", "Tensors"]),
    (
        "vedo.shapes.curves",
        [
            "Line",
            "DashedLine",
            "RoundedLine",
            "Tube",
            "Tubes",
            "ThickTube",
            "Lines",
            "Arc",
            "Spline",
            "KSpline",
            "CSpline",
            "Bezier",
            "NormalLines",
            "Ribbon",
            "Arrow",
            "Arrows",
            "Arrow2D",
            "Arrows2D",
            "FlatArrow",
        ],
    ),
    (
        "vedo.shapes.primitives",
        [
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
        ],
    ),
    ("vedo.shapes.markers", ["Marker", "Brace", "Star3D", "Cross3D", "ParametricShape"]),
    ("vedo.shapes.text", ["Text3D", "Text2D", "Latex", "_reps"]),
    ("vedo.shapes.analysis", ["ConvexHull"]),
)

__all__ = [*_LAZY_EXPORTS, "VedoLogo"]


def __getattr__(name):
    return getattr_lazy(__name__, globals(), name, attr_map=_LAZY_EXPORT_MAP)


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP)
