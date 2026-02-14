#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Interactive contour and point-cloud tools extracted from vedo.addons."""


import numpy as np
from typing_extensions import Self

import vedo
import vedo.vtkclasses as vtki

from vedo import utils
from vedo.colors import get_color
from vedo.pointcloud import Points


class PointCloudWidget:
    """
    Point cloud widget to interactively add or remove points in a point cloud.
    """

    def __init__(self, points):
        self.widget = vtki.new("PointCloudWidget")
        self.name = "PointCloudWidget"
        self.representation = self.widget.GetRepresentation()

        if utils.is_sequence(points):
            points = Points(points)
        self.points = points
        self.representation.PlacePointCloud(self.points.dataset)
        self.representation.BuildRepresentation()

    def add_to(self, plotter) -> "PointCloudWidget":
        """Add the point cloud widget to a Plotter instance."""
        if not isinstance(plotter, vedo.Plotter):
            vedo.logger.error("PointCloudWidget: add_to() requires a Plotter instance")
            return self
        self.widget.SetInteractor(plotter.interactor)
        self.representation.SetRenderer(plotter.renderer)
        self.widget.On()
        return self

    def on(self) -> "PointCloudWidget":
        """Activate/Enable the point cloud widget."""
        self.widget.On()
        self.widget.Render()
        return self

    def off(self) -> "PointCloudWidget":
        """Disactivate/Disable the point cloud widget."""
        self.widget.Off()
        self.widget.Render()
        return self

    @property
    def interactor(self):
        """Return the current interactor."""
        return self.GetInteractor()

    @interactor.setter
    def interactor(self, iren):
        """Set the current interactor."""
        self.widget.SetInteractor(iren)
        self.representation.SetInteractor(iren)

    @property
    def renderer(self):
        """Return the current renderer."""
        return self.representation.GetRenderer()

    @renderer.setter
    def renderer(self, ren):
        """Set the current renderer."""
        self.widget.SetRenderer(ren)
        self.widget.Render()

    def add_observer(self, event, func, priority=1) -> int:
        """Add an observer to the widget."""
        event = utils.get_vtk_name_event(event)
        cid = self.widget.AddObserver(event, func, priority)
        return cid

    def get_point_coordinates(self) -> np.ndarray:
        """Get the coordinates of a point by its index."""
        v = [0.0, 0.0, 0.0]
        self.representation.GetPointCoordinates(v)
        return np.array(v)


class SplineTool(vtki.vtkContourWidget):
    """
    Spline tool, draw a spline through a set of points interactively.
    """

    def __init__(
        self,
        points,
        pc="k",
        ps=8,
        lc="r4",
        ac="g5",
        lw=2,
        alpha=1,
        closed=False,
        ontop=True,
        can_add_nodes=True,
    ):
        super().__init__()

        self.name = "SplineTool"

        self.representation = self.GetRepresentation()
        self.representation.SetAlwaysOnTop(ontop)
        self.SetAllowNodePicking(can_add_nodes)

        self.representation.GetLinesProperty().SetColor(get_color(lc))
        self.representation.GetLinesProperty().SetLineWidth(lw)
        self.representation.GetLinesProperty().SetOpacity(alpha)
        if lw == 0 or alpha == 0:
            self.representation.GetLinesProperty().SetOpacity(0)

        self.representation.GetActiveProperty().SetLineWidth(lw + 1)
        self.representation.GetActiveProperty().SetColor(get_color(ac))

        self.representation.GetProperty().SetColor(get_color(pc))
        self.representation.GetProperty().SetPointSize(ps)
        self.representation.GetProperty().RenderPointsAsSpheresOn()
        self.SetRepresentation(self.representation)

        if utils.is_sequence(points):
            self.points = Points(points)
        else:
            self.points = points

        self.closed = closed

    @property
    def interactor(self):
        """Return the current interactor."""
        return self.GetInteractor()

    @interactor.setter
    def interactor(self, iren):
        """Set the current interactor."""
        self.SetInteractor(iren)

    def add_node(self, pt) -> "SplineTool":
        """Add one point at a specified position in world/display coordinates."""
        if len(pt) == 2:
            self.representation.AddNodeAtDisplayPosition(int(pt[0]), int(pt[1]))
        else:
            self.representation.AddNodeAtWorldPosition(pt)
        return self

    def add_observer(self, event, func, priority=1) -> int:
        """Add an observer to the widget."""
        event = utils.get_vtk_name_event(event)
        cid = self.AddObserver(event, func, priority)
        return cid

    def remove_node(self, i: int) -> "SplineTool":
        """Remove specific node by its index."""
        self.representation.DeleteNthNode(i)
        return self

    def on(self) -> "SplineTool":
        """Activate/Enable the tool."""
        self.On()
        self.Render()
        return self

    def off(self) -> "SplineTool":
        """Disactivate/Disable the tool."""
        self.Off()
        self.Render()
        return self

    def toggle(self) -> "SplineTool":
        """Toggle the visibility of the tool."""
        self.SetEnabled(not self.GetEnabled())
        return self

    def render(self) -> "SplineTool":
        """Render the spline."""
        self.Render()
        return self

    def lw(self, lw: int) -> "SplineTool":
        """Set the line width of the spline."""
        self.representation.GetLinesProperty().SetLineWidth(lw)
        self.representation.GetActiveProperty().SetLineWidth(lw)
        return self

    def ps(self, ps: int) -> "SplineTool":
        """Set the point size of the spline."""
        self.representation.GetProperty().SetPointSize(ps)
        return self

    def point_color(self, c: str | tuple) -> "SplineTool":
        """Set the color of the spline points."""
        c = get_color(c)
        self.representation.GetProperty().SetColor(c)
        return self

    def color(self, c: str | tuple) -> "SplineTool":
        """Set the color of the spline."""
        c = get_color(c)
        self.representation.GetProperty().SetColor(c)
        self.representation.GetLinesProperty().SetColor(c)
        self.representation.GetActiveProperty().SetColor(c)
        return self

    def closed_loop(self, value: bool) -> "SplineTool":
        """Set whether the spline is a closed loop."""
        self.closed = value
        self.representation.SetClosedLoop(value)
        return self

    def spline(self) -> vedo.Line:
        """Return the vedo.Line object."""
        self.representation.SetClosedLoop(self.closed)
        self.representation.BuildRepresentation()
        pd = self.representation.GetContourRepresentationAsPolyData()
        ln = vedo.Line(pd, lw=2, c="k")
        return ln

    def nodes(self, onscreen=False) -> np.ndarray:
        """Return the current spline nodes in world or screen coordinates."""
        n = self.representation.GetNumberOfNodes()
        pts = []
        for i in range(n):
            p = [0.0, 0.0, 0.0]
            if onscreen:
                self.representation.GetNthNodeDisplayPosition(i, p)
            else:
                self.representation.GetNthNodeWorldPosition(i, p)
            pts.append(p)
        return np.array(pts)

    def node_position(self, i, pt, onscreen=False) -> "SplineTool":
        """Set the position of a specific node by index."""
        n = self.representation.GetNumberOfNodes()
        if i < 0 or i >= n:
            vedo.logger.error(f"SplineTool: index {i} out of range [0-{n-1}]")
            return self
        if onscreen:
            self.representation.SetNthNodeDisplayPosition(i, pt[0], pt[1])
        else:
            self.representation.SetNthNodeWorldPosition(i, pt)
        return self

    def set_nodes(self, pts: np.ndarray | list) -> "SplineTool":
        """Set all spline nodes from an array/list of points."""
        if isinstance(pts, list):
            pts = np.array(pts)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        for i in range(len(pts)):
            self.representation.SetNthNodeWorldPosition(i, pts[i])
        return self
