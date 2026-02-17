#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Analytical/derived shapes extracted from vedo.shapes."""

import numpy as np
import vedo
import vedo.vtkclasses as vtki

from vedo import utils
from vedo.mesh import Mesh
from vedo.pointcloud import Points
class ConvexHull(Mesh):
    """
    Create the 2D/3D convex hull from a set of points.
    """

    def __init__(self, pts) -> None:
        """
        Create the 2D/3D convex hull from a set of input points or input Mesh.

        Examples:
            - [convex_hull.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/convex_hull.py)

                ![](https://vedo.embl.es/images/advanced/convexHull.png)
        """
        if utils.is_sequence(pts):
            pts = utils.make3d(pts).astype(float)
            mesh = Points(pts)
        else:
            mesh = pts
        apoly = mesh.clean().dataset

        # Create the convex hull of the pointcloud
        z0, z1 = mesh.zbounds()
        d = mesh.diagonal_size()
        if (z1 - z0) / d > 0.0001:
            delaunay = vtki.new("Delaunay3D")
            delaunay.SetInputData(apoly)
            delaunay.Update()
            surfaceFilter = vtki.new("DataSetSurfaceFilter")
            surfaceFilter.SetInputConnection(delaunay.GetOutputPort())
            surfaceFilter.Update()
            out = surfaceFilter.GetOutput()
        else:
            delaunay = vtki.new("Delaunay2D")
            delaunay.SetInputData(apoly)
            delaunay.Update()
            fe = vtki.new("FeatureEdges")
            fe.SetInputConnection(delaunay.GetOutputPort())
            fe.BoundaryEdgesOn()
            fe.Update()
            out = fe.GetOutput()

        super().__init__(out, c=mesh.color(), alpha=0.75)
        self.flat()
        self.name = "ConvexHull"


