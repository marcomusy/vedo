#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input normalization helpers shared by mesh-like classes."""

from __future__ import annotations

import os

import vedo.vtkclasses as vtki


def is_path_like(obj) -> bool:
    """Return True if obj can be treated as a filesystem path."""
    return isinstance(obj, (str, os.PathLike))


def as_path(pathlike) -> "str | bytes":
    """Convert a path-like object to string (or bytes) path."""
    return os.fspath(pathlike)


def as_dataset(obj):
    """Unwrap vedo-style objects exposing a `.dataset` attribute."""
    if hasattr(obj, "dataset"):
        return obj.dataset
    return obj


def geometry_filter_to_polydata(inputobj):
    """Convert a generic VTK dataset to vtkPolyData via vtkGeometryFilter."""
    dataset = as_dataset(inputobj)
    if dataset is None or not hasattr(dataset, "GetClassName"):
        raise TypeError(f"expected a VTK dataset, got {type(inputobj)}")
    gf = vtki.new("GeometryFilter")
    gf.SetInputData(dataset)
    gf.Update()
    return gf.GetOutput()


def points_polydata_from_dataset(inputobj):
    """
    Build a vtkPolyData containing only points and point-data arrays.

    Raises:
        TypeError: if input does not expose VTK-like point-data APIs.
    """
    dataset = as_dataset(inputobj)
    if not hasattr(dataset, "GetPoints"):
        raise TypeError("input object does not expose GetPoints()")

    vvpts = dataset.GetPoints()
    if vvpts is None:
        raise TypeError("input dataset has no points")

    poly = vtki.new("PolyData")
    poly.SetPoints(vvpts)

    pd = dataset.GetPointData()
    out_pd = poly.GetPointData()
    for i in range(pd.GetNumberOfArrays()):
        arr = pd.GetArray(i)
        if arr is not None:
            out_pd.AddArray(arr)

    carr = vtki.new("CellArray")
    for i in range(poly.GetNumberOfPoints()):
        carr.InsertNextCell(1)
        carr.InsertCellPoint(i)
    poly.SetVerts(carr)
    return poly
