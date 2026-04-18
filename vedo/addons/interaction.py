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

    def add_to(self, plotter) -> PointCloudWidget:
        """Add the point cloud widget to a Plotter instance."""
        if not isinstance(plotter, vedo.Plotter):
            vedo.logger.error("PointCloudWidget: add_to() requires a Plotter instance")
            return self
        self.widget.SetInteractor(plotter.interactor)
        self.representation.SetRenderer(plotter.renderer)
        self.widget.On()
        return self

    def on(self) -> PointCloudWidget:
        """Activate/Enable the point cloud widget."""
        self.widget.On()
        self.widget.Render()
        return self

    def off(self) -> PointCloudWidget:
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


class LineWidget:
    """
    An interactive widget to place and manipulate a 3D line segment.

    Two draggable sphere handles mark the endpoints. The line can be
    translated, scaled, and its endpoints repositioned interactively.

    Use `add_to(plotter)` to activate the widget in a scene.
    Read back `p1`/`p2` inside an observer callback to get updated positions.

    Example:
        ```python
        from vedo import Plotter, Sphere
        from vedo.addons import LineWidget

        def on_move(widget, event):
            print(widget.p1, "->", widget.p2, "  length:", round(widget.length, 3))

        plt = Plotter()
        plt += Sphere().alpha(0.3)
        lw = LineWidget((-0.5, 0, 0), (0.5, 0, 0))
        lw.add_to(plt)
        lw.add_observer("interaction", on_move)
        plt.show().close()
        ```
    """

    def __init__(
        self,
        p1=(-0.5, 0, 0),
        p2=(0.5, 0, 0),
        lc="k3",
        pc="k4",
        lw=2,
        ps=10,
        alpha=1.0,
        res=2,
    ):
        """
        Create an interactive line-segment widget.

        Args:
            p1 (list): world coordinates of the first endpoint.
            p2 (list): world coordinates of the second endpoint.
            lc (color): color of the line.
            pc (color): color of the endpoint handles.
            lw (int): line width in pixels.
            ps (int): handle sphere size (pixels).
            alpha (float): opacity of the line and handles.
            res (int): number of points along the line (including endpoints).
                `points` returns exactly `res` equally-spaced points.

        Examples:
            - [line_widget.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/line_widget.py)
        """
        self.name = "LineWidget"
        self.widget = vtki.new("LineWidget2")
        self.representation = vtki.new("LineRepresentation")
        self._callback_id = None

        self.representation.SetResolution(max(1, int(res) - 1))  # VTK: npts = res+1
        self.representation.SetPoint1WorldPosition(list(p1))
        self.representation.SetPoint2WorldPosition(list(p2))

        lc_ = get_color(lc)
        pc_ = get_color(pc)

        lp = self.representation.GetLineProperty()
        lp.SetColor(lc_)
        lp.SetLineWidth(lw)
        lp.SetOpacity(alpha)
        lp.LightingOff()

        for ep_prop in (
            self.representation.GetEndPointProperty(),
            self.representation.GetEndPoint2Property(),
        ):
            ep_prop.SetColor(pc_)
            ep_prop.SetOpacity(alpha)
            ep_prop.RenderPointsAsSpheresOn()

        for sel_prop in (
            self.representation.GetSelectedEndPointProperty(),
            self.representation.GetSelectedEndPoint2Property(),
        ):
            sel_prop.SetColor(get_color("red3"))
            sel_prop.SetOpacity(alpha)
            sel_prop.RenderPointsAsSpheresOn()

        self.representation.SetHandleSize(ps)
        self.widget.SetRepresentation(self.representation)

    # ── geometry properties ───────────────────────────────────────────────

    @property
    def p1(self) -> np.ndarray:
        """World position of the first endpoint."""
        return np.array(self.representation.GetPoint1WorldPosition())

    @p1.setter
    def p1(self, value) -> None:
        self.representation.SetPoint1WorldPosition(list(value))

    @property
    def p2(self) -> np.ndarray:
        """World position of the second endpoint."""
        return np.array(self.representation.GetPoint2WorldPosition())

    @p2.setter
    def p2(self, value) -> None:
        self.representation.SetPoint2WorldPosition(list(value))

    @property
    def midpoint(self) -> np.ndarray:
        """Midpoint of the line segment."""
        return (self.p1 + self.p2) / 2

    @property
    def length(self) -> float:
        """Length of the line segment."""
        return float(np.linalg.norm(self.p2 - self.p1))

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from p1 to p2. Returns zeros if the two points coincide."""
        d = self.p2 - self.p1
        n = np.linalg.norm(d)
        return d / n if n > 0 else d

    @property
    def points(self) -> np.ndarray:
        """
        Return `res` equally-spaced points along the line segment as a (res, 3)
        numpy array. `res` is the value set at construction time.
        """
        self.representation.BuildRepresentation()
        pd = vtki.new("PolyData")
        self.representation.GetPolyData(pd)
        return utils.vtk2numpy(pd.GetPoints().GetData())

    # ── visual styling ────────────────────────────────────────────────────

    def line_color(self, c) -> "LineWidget":
        """Set the color of the line."""
        self.representation.GetLineProperty().SetColor(get_color(c))
        return self

    def handle_color(self, c) -> "LineWidget":
        """Set the color of both endpoint handles."""
        c_ = get_color(c)
        self.representation.GetEndPointProperty().SetColor(c_)
        self.representation.GetEndPoint2Property().SetColor(c_)
        return self

    def lw(self, value: int) -> "LineWidget":
        """Set the line width in pixels."""
        self.representation.GetLineProperty().SetLineWidth(value)
        return self

    def ps(self, value: int) -> "LineWidget":
        """Set the handle sphere size in pixels."""
        self.representation.SetHandleSize(value)
        return self

    def show_distance(self, value=True, fmt="{:.3g}") -> "LineWidget":
        """
        Show or hide the distance annotation along the line.

        Args:
            value (bool): whether to show the annotation.
            fmt (str): Python format string used for the numeric label.
        """
        self.representation.SetDistanceAnnotationVisibility(value)
        if value:
            self.representation.SetDistanceAnnotationFormat(
                fmt.replace("{:.3g}", "%-#6.3g")
                   .replace("{:.2f}", "%-#6.2f")
                   .replace("{:.1f}", "%-#6.1f")
            )
        return self

    # ── lifecycle ─────────────────────────────────────────────────────────

    def add_to(self, plt) -> "LineWidget":
        """
        Add the widget to a `Plotter` instance and enable it.

        Args:
            plt (Plotter): the target plotter.
        """
        _p1 = self.p1.copy()
        _p2 = self.p2.copy()
        self.widget.SetInteractor(plt.interactor)
        self.widget.SetCurrentRenderer(plt.renderer)
        self.representation.PlaceWidget(plt.renderer.ComputeVisiblePropBounds())
        # PlaceWidget resets endpoint positions — restore the user-specified ones
        self.representation.SetPoint1WorldPosition(list(_p1))
        self.representation.SetPoint2WorldPosition(list(_p2))
        self.widget.On()
        if self.widget not in plt.widgets:
            plt.widgets.append(self.widget)
        return self

    def remove_from(self, plt) -> "LineWidget":
        """Remove the widget from a `Plotter` instance."""
        self.widget.Off()
        if self.widget in plt.widgets:
            plt.widgets.remove(self.widget)
        return self

    def on(self) -> "LineWidget":
        """Enable the widget."""
        self.widget.On()
        return self

    def off(self) -> "LineWidget":
        """Disable the widget."""
        self.widget.Off()
        return self

    def toggle(self) -> "LineWidget":
        """Toggle the widget on/off."""
        if self.widget.GetEnabled():
            self.widget.Off()
        else:
            self.widget.On()
        return self

    def is_enabled(self) -> bool:
        """Return True if the widget is currently enabled."""
        return bool(self.widget.GetEnabled())

    # ── observers ─────────────────────────────────────────────────────────

    def add_observer(self, event, func, priority=1) -> int:
        """
        Add an observer callback to the widget.

        The callback receives `(widget, event_name)` and can read
        `widget.p1`, `widget.p2`, `widget.length`, etc.

        Args:
            event (str): event name, e.g. `"interaction"`, `"start"`, `"end"`.
            func (callable): callback function.
            priority (int): observer priority.

        Returns:
            int: observer id (use with `remove_observer`).
        """
        event = utils.get_vtk_name_event(event)
        return self.widget.AddObserver(event, func, priority)

    def remove_observer(self, cid: int) -> "LineWidget":
        """Remove a specific observer by its id."""
        self.widget.RemoveObserver(cid)
        return self

    def remove_observers(self, event="") -> "LineWidget":
        """Remove all observers, or only those for a specific event."""
        if not event:
            self.widget.RemoveAllObservers()
        else:
            self.widget.RemoveObservers(utils.get_vtk_name_event(event))
        return self

    # ── output ────────────────────────────────────────────────────────────

    def get_line(self) -> "vedo.Line":
        """
        Return the current widget state as a `vedo.Line` object.

        Useful for extracting geometry after interaction.
        """
        pd = vtki.vtkPolyData()
        self.representation.GetPolyData(pd)
        return vedo.Line(self.p1, self.p2)

    def __repr__(self) -> str:
        return (
            f"LineWidget(p1={self.p1.tolist()}, p2={self.p2.tolist()}, "
            f"length={self.length:.4g})"
        )


class PointWidget:
    """
    An interactive widget to place a single draggable 3D point.

    Use `add_to(plotter)` to activate the widget in a scene.
    Read back `pos` inside an observer callback to get the current position.

    Example:
        ```python
        from vedo import Plotter, Sphere
        from vedo.addons import PointWidget

        def on_move(widget, event):
            print(widget.pos)

        plt = Plotter()
        plt += Sphere().alpha(0.3)
        pw = PointWidget((0.5, 0, 0))
        pw.add_to(plt)
        pw.add_observer("interaction", on_move)
        plt.show().close()
        ```
    """

    def __init__(
        self,
        pos=(0, 0, 0),
        c="red5",
        alpha=1.0,
        ps=0.1,
    ):
        """
        Create an interactive single-point widget.

        Args:
            pos (list): initial world coordinates of the point.
            c (color): color of the point handle.
            alpha (float): opacity of the point handle.
            ps (float): sphere radius of the point handle in world units.
        """
        self.name = "PointWidget"
        self.widget = vtki.new("HandleWidget")
        self.representation = vtki.new("SphereHandleRepresentation")
        self._pos = list(pos)

        self.representation.SetWorldPosition(list(pos))
        self.representation.SetSphereRadius(ps * 0.05)
        self.representation.GetProperty().SetColor(get_color(c))
        self.representation.GetProperty().SetOpacity(alpha)
        self.representation.GetSelectedProperty().SetColor(get_color("red3"))
        self.representation.GetSelectedProperty().SetOpacity(alpha)
        self.widget.SetRepresentation(self.representation)

    # ── geometry ──────────────────────────────────────────────────────────

    @property
    def pos(self) -> np.ndarray:
        """World position of the point."""
        return np.array(self.representation.GetWorldPosition())

    @pos.setter
    def pos(self, value) -> None:
        self._pos = list(value)
        self.representation.SetWorldPosition(list(value))

    # ── visual styling ────────────────────────────────────────────────────

    def color(self, c) -> "PointWidget":
        """Set the point color."""
        self.representation.GetProperty().SetColor(get_color(c))
        return self

    def alpha(self, value: float) -> "PointWidget":
        """Set the point opacity."""
        self.representation.GetProperty().SetOpacity(value)
        return self

    def ps(self, value: float) -> "PointWidget":
        """Set the sphere radius of the point handle."""
        self.representation.SetSphereRadius(value)
        return self

    # ── lifecycle ─────────────────────────────────────────────────────────

    def add_to(self, plt) -> "PointWidget":
        """Add the widget to a `Plotter` instance and enable it."""
        self.widget.SetInteractor(plt.interactor)
        self.widget.SetCurrentRenderer(plt.renderer)
        self.representation.SetWorldPosition(self._pos)
        self.widget.On()
        if self.widget not in plt.widgets:
            plt.widgets.append(self.widget)
        return self

    def remove_from(self, plt) -> "PointWidget":
        """Remove the widget from a `Plotter` instance."""
        self.widget.Off()
        if self.widget in plt.widgets:
            plt.widgets.remove(self.widget)
        return self

    def on(self) -> "PointWidget":
        """Enable the widget."""
        self.widget.On()
        return self

    def off(self) -> "PointWidget":
        """Disable the widget."""
        self.widget.Off()
        return self

    def toggle(self) -> "PointWidget":
        """Toggle the widget on/off."""
        if self.widget.GetEnabled():
            self.widget.Off()
        else:
            self.widget.On()
        return self

    def is_enabled(self) -> bool:
        """Return True if the widget is currently enabled."""
        return bool(self.widget.GetEnabled())

    # ── observers ─────────────────────────────────────────────────────────

    def add_observer(self, event, func, priority=1) -> int:
        """Add an observer callback for a widget event."""
        event = utils.get_vtk_name_event(event)
        return self.widget.AddObserver(event, func, priority)

    def remove_observer(self, cid: int) -> "PointWidget":
        """Remove an observer by its callback id."""
        self.widget.RemoveObserver(cid)
        return self

    def remove_observers(self, event="") -> "PointWidget":
        """Remove all observers, or those for a specific event."""
        if event:
            self.widget.RemoveObservers(utils.get_vtk_name_event(event))
        else:
            self.widget.RemoveAllObservers()
        return self

    def __repr__(self) -> str:
        return f"PointWidget(pos={self.pos.tolist()})"


class SphereWidget:
    """
    An interactive widget to place and resize a 3D sphere.

    A draggable handle on the sphere surface lets you resize interactively.
    The sphere body can be translated by clicking and dragging.

    Use `add_to(plotter)` to activate the widget in a scene.
    Read back `center` and `radius` inside an observer callback.

    Example:
        ```python
        from vedo import Plotter
        from vedo.addons import SphereWidget

        def on_move(widget, event):
            print(widget.center, widget.radius)

        plt = Plotter()
        sw = SphereWidget(center=(0, 0, 0), r=1)
        sw.add_to(plt)
        sw.add_observer("interaction", on_move)
        plt.show().close()
        ```
    """

    def __init__(
        self,
        center=(0, 0, 0),
        r=1.0,
        c="green5",
        alpha=0.5,
        res=24,
    ):
        """
        Create an interactive sphere widget.

        Args:
            center (list): world coordinates of the sphere center.
            r (float): radius of the sphere.
            c (color): color of the sphere surface.
            alpha (float): opacity of the sphere surface.
            res (int): sphere surface resolution (phi and theta subdivisions).
        """
        self.name = "SphereWidget"
        self.widget = vtki.new("SphereWidget2")
        self.representation = vtki.new("SphereRepresentation")

        self.representation.SetCenter(list(center))
        self.representation.SetRadius(r)
        self.representation.SetPhiResolution(res)
        self.representation.SetThetaResolution(res*2)
        self.representation.SetRepresentationToSurface()

        c_ = get_color(c)
        sp = self.representation.GetSphereProperty()
        sp.SetColor(c_)
        sp.SetOpacity(alpha)

        sp_sel = self.representation.GetSelectedSphereProperty()
        sp_sel.SetColor(get_color("red3"))
        sp_sel.SetOpacity(min(alpha + 0.2, 1.0))

        hp = self.representation.GetHandleProperty()
        hp.SetColor(get_color("white"))
        hp.RenderPointsAsSpheresOn()

        hp_sel = self.representation.GetSelectedHandleProperty()
        hp_sel.SetColor(get_color("red3"))
        hp_sel.RenderPointsAsSpheresOn()

        self.widget.SetRepresentation(self.representation)

    # ── geometry properties ───────────────────────────────────────────────

    @property
    def center(self) -> np.ndarray:
        """World position of the sphere center."""
        return np.array(self.representation.GetCenter())

    @center.setter
    def center(self, value) -> None:
        self.representation.SetCenter(list(value))

    @property
    def radius(self) -> float:
        """Radius of the sphere."""
        return float(self.representation.GetRadius())

    @radius.setter
    def radius(self, value) -> None:
        self.representation.SetRadius(float(value))

    @property
    def points(self) -> np.ndarray:
        """
        Return the surface point coordinates of the sphere as a (N, 3) array.
        N is determined by the `res` value set at construction time.
        Useful as seed points for streamline integration.
        """
        self.representation.BuildRepresentation()
        pd = vtki.new("PolyData")
        self.representation.GetPolyData(pd)
        return utils.vtk2numpy(pd.GetPoints().GetData())

    # ── visual styling ────────────────────────────────────────────────────

    def color(self, c) -> "SphereWidget":
        """Set the sphere surface color."""
        self.representation.GetSphereProperty().SetColor(get_color(c))
        return self

    def alpha(self, value: float) -> "SphereWidget":
        """Set the sphere surface opacity."""
        self.representation.GetSphereProperty().SetOpacity(value)
        return self

    def lw(self, value: int) -> "SphereWidget":
        """Set the wireframe line width (when in wireframe mode)."""
        self.representation.GetSphereProperty().SetLineWidth(value)
        return self

    # ── lifecycle ─────────────────────────────────────────────────────────

    def add_to(self, plt) -> "SphereWidget":
        """Add the widget to a `Plotter` instance and enable it."""
        c, r = self.center.copy(), self.radius
        self.widget.SetInteractor(plt.interactor)
        self.widget.SetCurrentRenderer(plt.renderer)
        self.representation.PlaceWidget(plt.renderer.ComputeVisiblePropBounds())
        # Restore center/radius — PlaceWidget resets them
        self.representation.SetCenter(list(c))
        self.representation.SetRadius(r)
        self.widget.On()
        if self.widget not in plt.widgets:
            plt.widgets.append(self.widget)
        return self

    def remove_from(self, plt) -> "SphereWidget":
        """Remove the widget from a `Plotter` instance."""
        self.widget.Off()
        if self.widget in plt.widgets:
            plt.widgets.remove(self.widget)
        return self

    def on(self) -> "SphereWidget":
        """Enable the widget."""
        self.widget.On()
        return self

    def off(self) -> "SphereWidget":
        """Disable the widget."""
        self.widget.Off()
        return self

    def toggle(self) -> "SphereWidget":
        """Toggle the widget on/off."""
        if self.widget.GetEnabled():
            self.widget.Off()
        else:
            self.widget.On()
        return self

    def is_enabled(self) -> bool:
        """Return True if the widget is currently enabled."""
        return bool(self.widget.GetEnabled())

    # ── observers ─────────────────────────────────────────────────────────

    def add_observer(self, event, func, priority=1) -> int:
        """Add an observer callback for a widget event."""
        event = utils.get_vtk_name_event(event)
        return self.widget.AddObserver(event, func, priority)

    def remove_observer(self, cid: int) -> "SphereWidget":
        """Remove an observer by its callback id."""
        self.widget.RemoveObserver(cid)
        return self

    def remove_observers(self, event="") -> "SphereWidget":
        """Remove all observers, or those for a specific event."""
        if event:
            self.widget.RemoveObservers(utils.get_vtk_name_event(event))
        else:
            self.widget.RemoveAllObservers()
        return self

    # ── output ────────────────────────────────────────────────────────────

    def get_sphere(self) -> "vedo.Sphere":
        """Return the current sphere as a `vedo.Sphere` object."""
        return vedo.shapes.Sphere(self.center, r=self.radius)

    def __repr__(self) -> str:
        return (
            f"SphereWidget(center={self.center.tolist()}, radius={self.radius:.4g})"
        )


class CylinderWidget:
    """
    An interactive widget to place and resize a 3D cylinder.

    The cylinder axis, center, and radius can be manipulated interactively.
    Use `add_to(plotter)` to activate the widget in a scene.
    Read back `center`, `radius`, `axis`, and `points` in an observer callback.

    Example:
        ```python
        from vedo import Plotter, UnstructuredGrid, dataurl
        from vedo.addons import CylinderWidget

        def on_move(widget, event):
            print(cw.center, cw.radius, cw.axis)

        domain = UnstructuredGrid(dataurl + "comb_domain.vtu")
        plt = Plotter()
        cw = CylinderWidget((5, 0, 30), domain, r=2, axis=(0, 0, 1))
        cw.add_to(plt)
        cw.add_observer("interaction", on_move)
        plt.show().close()
        ```
    """

    def __init__(
        self,
        center,
        bounds,
        r=1.0,
        axis=(1, 0, 0),
        c="green5",
        alpha=0.3,
        res=12,
    ):
        """
        Create an interactive cylinder widget.

        Args:
            center (list): world coordinates of the cylinder center.
            bounds (list, vedo object): placement bounds as `[xmin,xmax, ymin,ymax, zmin,zmax]`
                or any vedo object whose `.bounds()` is used.
            r (float): cylinder radius.
            axis (list): unit vector along the cylinder axis.
            c (color): color of the cylinder surface.
            alpha (float): opacity of the cylinder surface.
            res (int, tuple): resolution as `(radial, axial)` or a single int for both.
        """
        self.name = "CylinderWidget"
        self.widget = vtki.new("ImplicitCylinderWidget")
        self.representation = vtki.new("ImplicitCylinderRepresentation")
        if utils.is_sequence(res):
            self._res_r, self._res_a = int(res[0]), int(res[1])
        else:
            self._res_r = self._res_a = int(res)

        self.widget.SetRepresentation(self.representation)
        self.representation.DrawCylinderOn()
        self.representation.GetOutlineProperty().SetOpacity(0)
        self.representation.GetSelectedOutlineProperty().SetOpacity(0)
        self.representation.GetEdgesProperty().SetOpacity(0)
        self.representation.OutlineTranslationOff()

        c_ = get_color(c)
        cp = self.representation.GetCylinderProperty()
        cp.SetColor(c_)
        cp.SetOpacity(alpha)

        sp = self.representation.GetSelectedCylinderProperty() if hasattr(
            self.representation, "GetSelectedCylinderProperty") else None
        if sp:
            sp.SetColor(get_color("red3"))
            sp.SetOpacity(min(alpha + 0.2, 1.0))

        ap = self.representation.GetAxisProperty()
        ap.SetColor(c_)
        ap.SetLineWidth(2)

        if hasattr(bounds, "bounds"):
            bounds = bounds.bounds()
        bounds = np.asarray(bounds, dtype=float)
        # Use tight bounds around the cylinder so the visual is not clipped.
        # Half-length is derived from the domain diagonal so the cylinder
        # reaches across the scene naturally.
        c = np.asarray(center, dtype=float)
        half_len = np.linalg.norm(bounds[1::2] - bounds[::2]) / 2
        pad = max(float(r), half_len)
        tight = [c[0]-pad, c[0]+pad, c[1]-pad, c[1]+pad, c[2]-pad, c[2]+pad]
        self.representation.PlaceWidget(tight)
        self.representation.SetCenter(list(c))
        self.representation.SetAxis(list(axis))
        self.representation.SetRadius(float(r))

    # ── geometry properties ───────────────────────────────────────────────

    @property
    def center(self) -> np.ndarray:
        """World position of the cylinder center."""
        return np.array(self.representation.GetCenter())

    @center.setter
    def center(self, value) -> None:
        self.representation.SetCenter(list(value))

    @property
    def radius(self) -> float:
        """Radius of the cylinder."""
        return float(self.representation.GetRadius())

    @radius.setter
    def radius(self, value) -> None:
        self.representation.SetRadius(float(value))

    @property
    def axis(self) -> np.ndarray:
        """Unit vector along the cylinder axis."""
        a = np.array(self.representation.GetAxis())
        n = np.linalg.norm(a)
        return a / n if n > 0 else a

    @axis.setter
    def axis(self, value) -> None:
        self.representation.SetAxis(list(value))

    @property
    def length(self) -> float:
        """Cylinder length derived from the widget placement bounds."""
        b = np.array(self.representation.GetWidgetBounds())
        ax = self.axis
        corners = np.array([[b[i], b[j], b[k]]
            for i in [0, 1] for j in [2, 3] for k in [4, 5]])
        projs = corners @ ax
        return float(projs.max() - projs.min())

    @property
    def points(self) -> np.ndarray:
        """
        Return surface points of the cylinder as a (N, 3) array.
        Points are sampled with `res` circumferential divisions and 2 rows
        (top/bottom caps), derived analytically from center, axis, and radius.
        """
        ax = self.axis
        half = self.length / 2
        perp = np.array([1, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1, 0])
        e1 = np.cross(ax, perp); e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)

        u = np.linspace(0, 2 * np.pi, self._res_r, endpoint=False)
        v = np.linspace(-half, half, self._res_a)
        uu, vv = np.meshgrid(u, v)

        pts = (
            self.center
            + self.radius * (np.cos(uu)[..., None] * e1 + np.sin(uu)[..., None] * e2)
            + vv[..., None] * ax
        )
        return pts.reshape(-1, 3)

    # ── visual styling ────────────────────────────────────────────────────

    def color(self, c) -> "CylinderWidget":
        """Set the cylinder surface color."""
        self.representation.GetCylinderProperty().SetColor(get_color(c))
        return self

    def alpha(self, value: float) -> "CylinderWidget":
        """Set the cylinder surface opacity."""
        self.representation.GetCylinderProperty().SetOpacity(value)
        return self

    # ── lifecycle ─────────────────────────────────────────────────────────

    def add_to(self, plt) -> "CylinderWidget":
        """Add the widget to a `Plotter` instance and enable it."""
        self.widget.SetInteractor(plt.interactor)
        self.widget.SetCurrentRenderer(plt.renderer)
        self.widget.On()
        if self.widget not in plt.widgets:
            plt.widgets.append(self.widget)
        return self

    def remove_from(self, plt) -> "CylinderWidget":
        """Remove the widget from a `Plotter` instance."""
        self.widget.Off()
        if self.widget in plt.widgets:
            plt.widgets.remove(self.widget)
        return self

    def on(self) -> "CylinderWidget":
        """Enable the widget."""
        self.widget.On()
        return self

    def off(self) -> "CylinderWidget":
        """Disable the widget."""
        self.widget.Off()
        return self

    def toggle(self) -> "CylinderWidget":
        """Toggle the widget on/off."""
        if self.widget.GetEnabled():
            self.widget.Off()
        else:
            self.widget.On()
        return self

    def is_enabled(self) -> bool:
        """Return True if the widget is currently enabled."""
        return bool(self.widget.GetEnabled())

    # ── observers ─────────────────────────────────────────────────────────

    def add_observer(self, event, func, priority=1) -> int:
        """Add an observer callback for a widget event."""
        event = utils.get_vtk_name_event(event)
        return self.widget.AddObserver(event, func, priority)

    def remove_observer(self, cid: int) -> "CylinderWidget":
        """Remove an observer by its callback id."""
        self.widget.RemoveObserver(cid)
        return self

    def remove_observers(self, event="") -> "CylinderWidget":
        """Remove all observers, or those for a specific event."""
        if event:
            self.widget.RemoveObservers(utils.get_vtk_name_event(event))
        else:
            self.widget.RemoveAllObservers()
        return self

    def __repr__(self) -> str:
        return (
            f"CylinderWidget(center={self.center.tolist()}, "
            f"radius={self.radius:.4g}, axis={self.axis.tolist()})"
        )


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

    def add_node(self, pt) -> SplineTool:
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

    def remove_node(self, i: int) -> SplineTool:
        """Remove specific node by its index."""
        self.representation.DeleteNthNode(i)
        return self

    def on(self) -> SplineTool:
        """Activate/Enable the tool."""
        self.On()
        self.Render()
        return self

    def off(self) -> SplineTool:
        """Disactivate/Disable the tool."""
        self.Off()
        self.Render()
        return self

    def toggle(self) -> SplineTool:
        """Toggle the visibility of the tool."""
        self.SetEnabled(not self.GetEnabled())
        return self

    def render(self) -> SplineTool:
        """Render the spline."""
        self.Render()
        return self

    def lw(self, lw: int) -> SplineTool:
        """Set the line width of the spline."""
        self.representation.GetLinesProperty().SetLineWidth(lw)
        self.representation.GetActiveProperty().SetLineWidth(lw)
        return self

    def ps(self, ps: int) -> SplineTool:
        """Set the point size of the spline."""
        self.representation.GetProperty().SetPointSize(ps)
        return self

    def point_color(self, c: str | tuple) -> SplineTool:
        """Set the color of the spline points."""
        c = get_color(c)
        self.representation.GetProperty().SetColor(c)
        return self

    def color(self, c: str | tuple) -> SplineTool:
        """Set the color of the spline."""
        c = get_color(c)
        self.representation.GetProperty().SetColor(c)
        self.representation.GetLinesProperty().SetColor(c)
        self.representation.GetActiveProperty().SetColor(c)
        return self

    def closed_loop(self, value: bool) -> SplineTool:
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

    def node_position(self, i, pt, onscreen=False) -> SplineTool:
        """Set the position of a specific node by index."""
        n = self.representation.GetNumberOfNodes()
        if i < 0 or i >= n:
            vedo.logger.error(f"SplineTool: index {i} out of range [0-{n - 1}]")
            return self
        if onscreen:
            self.representation.SetNthNodeDisplayPosition(i, pt[0], pt[1])
        else:
            self.representation.SetNthNodeWorldPosition(i, pt)
        return self

    def set_nodes(self, pts: np.ndarray | list) -> SplineTool:
        """Set all spline nodes from an array/list of points."""
        if isinstance(pts, list):
            pts = np.array(pts)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        for i in range(len(pts)):
            self.representation.SetNthNodeWorldPosition(i, pts[i])
        return self
