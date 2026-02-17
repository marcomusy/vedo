#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Pointcloud package facade."""

from .core import Point, Points
from .fits import (
    merge,
    fit_line,
    fit_circle,
    fit_plane,
    fit_sphere,
    pca_ellipse,
    pca_ellipsoid,
    project_point_on_variety,
)

__all__ = [
    "Points",
    "Point",
    "merge",
    "fit_line",
    "fit_circle",
    "fit_plane",
    "fit_sphere",
    "pca_ellipse",
    "pca_ellipsoid",
    "project_point_on_variety",
]
