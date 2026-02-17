#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Measurement helpers extracted from vedo.addons."""

from typing_extensions import Self
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import settings
from vedo import utils
from vedo import shapes
from vedo.colors import get_color
from vedo.assembly import Group
from vedo.mesh import Mesh
from vedo.pointcloud import merge
class Ruler2D(vtki.vtkAxisActor2D):
    """
    Create a ruler with tick marks, labels and a title.
    """

    def __init__(
        self,
        lw=2,
        ticks=True,
        labels=False,
        c="k",
        alpha=1,
        title="",
        font="Calco",
        font_size=24,
        bc=None,
    ):
        """
        Create a ruler with tick marks, labels and a title.

        Ruler2D is a 2D actor; that is, it is drawn on the overlay
        plane and is not occluded by 3D geometry.
        To use this class, specify two points defining the start and end
        with update_points() as 3D points.

        This class decides decides how to create reasonable tick
        marks and labels.

        Labels are drawn on the "right" side of the axis.
        The "right" side is the side of the axis on the right.
        The way the labels and title line up with the axis and tick marks
        depends on whether the line is considered horizontal or vertical.

        Arguments:
            lw : (int)
                width of the line in pixel units
            ticks : (bool)
                control if drawing the tick marks
            labels : (bool)
                control if drawing the numeric labels
            c : (color)
                color of the object
            alpha : (float)
                opacity of the object
            title : (str)
                title of the ruler
            font : (str)
                font face name. Check [available fonts here](https://vedo.embl.es/fonts).
            font_size : (int)
                font size
            bc : (color)
                background color of the title

        Example:
            ```python
            from vedo  import *
            plt = Plotter(axes=1, interactive=False)
            plt.show(Cube())
            rul = Ruler2D()
            rul.set_points([0,0,0], [0.5,0.5,0.5])
            plt.add(rul)
            plt.interactive().close()
            ```
            ![](https://vedo.embl.es/images/feats/dist_tool.png)
        """
        super().__init__()
        self.name = "Ruler2D"

        plt = vedo.current_plotter()
        if not plt:
            vedo.logger.error("Ruler2D need to initialize Plotter first.")
            raise RuntimeError()

        self.p0 = [0, 0, 0]
        self.p1 = [0, 0, 0]
        self.distance = 0
        self.title = title

        prop = self.GetProperty()
        tprop = self.GetTitleTextProperty()

        self.SetTitle(title)
        self.SetNumberOfLabels(9)

        if not font:
            font = settings.default_font
        if font.lower() == "courier":
            tprop.SetFontFamilyToCourier()
        elif font.lower() == "times":
            tprop.SetFontFamilyToTimes()
        elif font.lower() == "arial":
            tprop.SetFontFamilyToArial()
        else:
            tprop.SetFontFamily(vtki.VTK_FONT_FILE)
            tprop.SetFontFile(utils.get_font_path(font))
        tprop.SetFontSize(font_size)
        tprop.BoldOff()
        tprop.ItalicOff()
        tprop.ShadowOff()
        tprop.SetColor(get_color(c))
        tprop.SetOpacity(alpha)
        if bc is not None:
            bc = get_color(bc)
            tprop.SetBackgroundColor(bc)
            tprop.SetBackgroundOpacity(alpha)

        lprop = vtki.vtkTextProperty()
        lprop.ShallowCopy(tprop)
        self.SetLabelTextProperty(lprop)

        self.SetLabelFormat("%0.3g")
        self.SetTickVisibility(ticks)
        self.SetLabelVisibility(labels)
        prop.SetLineWidth(lw)
        prop.SetColor(get_color(c))

        self.renderer = plt.renderer
        self.cid = plt.interactor.AddObserver("RenderEvent", self._update_viz, 1.0)

    def color(self, c) -> Self:
        """Assign a new color."""
        c = get_color(c)
        self.GetTitleTextProperty().SetColor(c)
        self.GetLabelTextProperty().SetColor(c)
        self.GetProperty().SetColor(c)
        return self

    def off(self) -> None:
        """Switch off the ruler completely."""
        self.renderer.RemoveObserver(self.cid)
        self.renderer.RemoveActor(self)

    def set_points(self, p0, p1) -> Self:
        """Set new values for the ruler start and end points."""
        self.p0 = np.asarray(p0)
        self.p1 = np.asarray(p1)
        self._update_viz(0, 0)
        return self

    def _update_viz(self, _evt, _name) -> None:
        ren = self.renderer
        view_size = np.array(ren.GetSize())

        ren.SetWorldPoint(*self.p0, 1)
        ren.WorldToDisplay()
        disp_point1 = ren.GetDisplayPoint()[:2]
        disp_point1 = np.array(disp_point1) / view_size

        ren.SetWorldPoint(*self.p1, 1)
        ren.WorldToDisplay()
        disp_point2 = ren.GetDisplayPoint()[:2]
        disp_point2 = np.array(disp_point2) / view_size

        self.SetPoint1(*disp_point1)
        self.SetPoint2(*disp_point2)
        self.distance = np.linalg.norm(self.p1 - self.p0)
        self.SetRange(0.0, float(self.distance))
        if not self.title:
            self.SetTitle(utils.precision(self.distance, 3))


#####################################################################
class DistanceTool(Group):
    """
    Create a tool to measure the distance between two clicked points.
    """

    def __init__(self, plotter=None, c="k", lw=2):
        """
        Create a tool to measure the distance between two clicked points.

        Example:
            ```python
            from vedo import *
            mesh = ParametricShape("RandomHills").c("red5")
            plt = Plotter(axes=1)
            dtool = DistanceTool()
            dtool.on()
            plt.show(mesh, dtool)
            dtool.off()
            ```
            ![](https://vedo.embl.es/images/feats/dist_tool.png)
        """
        super().__init__()
        self.name = "DistanceTool"

        self.p0 = [0, 0, 0]
        self.p1 = [0, 0, 0]
        self.distance = 0
        if plotter is None:
            plotter = vedo.current_plotter()
        self.plotter = plotter
        # define self.callback as callable function:
        self.callback = lambda x: None
        self.cid = None
        self.color = c
        self.linewidth = lw
        self.toggle = True
        self.ruler = None
        self.title = ""

    def on(self) -> Self:
        """Switch tool on."""
        self.cid = self.plotter.add_callback("click", self._onclick)
        self.VisibilityOn()
        self.plotter.render()
        return self

    def off(self) -> None:
        """Switch tool off."""
        self.plotter.remove_callback(self.cid)
        self.VisibilityOff()
        self.ruler.off()
        self.plotter.render()

    def _onclick(self, event):
        if not event.actor:
            return

        self.clear()

        acts = []
        if self.toggle:
            self.p0 = event.picked3d
            acts.append(Point(self.p0, c=self.color))
        else:
            self.p1 = event.picked3d
            self.distance = np.linalg.norm(self.p1 - self.p0)
            acts.append(Point(self.p0, c=self.color))
            acts.append(Point(self.p1, c=self.color))
            self.ruler = Ruler2D(c=self.color)
            self.ruler.set_points(self.p0, self.p1)
            acts.append(self.ruler)

            if self.callback is not None:
                self.callback(event)

        for a in acts:
            try:
                self += a.actor
            except AttributeError:
                self += a
        self.toggle = not self.toggle


#####################################################################


def compute_visible_bounds(objs=None) -> list:
    """Calculate max objects bounds and sizes."""
    bns = []

    plt = vedo.current_plotter()
    if objs is None and plt:
        objs = plt.actors
    if callable(objs):
        objs = objs()
    elif not utils.is_sequence(objs):
        objs = [objs]

    actors = [ob.actor for ob in objs if hasattr(ob, "actor") and ob.actor]

    try:
        # this block fails for VolumeSlice as vtkImageSlice.GetBounds() returns a pointer..
        # in any case we dont need axes for that one.
        for a in actors:
            if a and a.GetUseBounds():
                b = a.GetBounds()
                if b:
                    bns.append(b)
        if bns:
            max_bns = np.max(bns, axis=0)
            min_bns = np.min(bns, axis=0)
            vbb = [min_bns[0], max_bns[1], min_bns[2], max_bns[3], min_bns[4], max_bns[5]]
        elif plt and plt.renderer:
            vbb = list(plt.renderer.ComputeVisiblePropBounds())
            max_bns = vbb
            min_bns = vbb
        sizes = np.array(
            [max_bns[1] - min_bns[0], max_bns[3] - min_bns[2], max_bns[5] - min_bns[4]]
        )
        return [vbb, sizes, min_bns, max_bns]

    except:
        return [[0, 0, 0, 0, 0, 0], [0, 0, 0], 0, 0]


#####################################################################
def Ruler3D(
    p1,
    p2,
    units_scale=1,
    label="",
    s=None,
    font=None,
    italic=0,
    prefix="",
    units="",  # eg.'μm'
    c=(0.2, 0.1, 0.1),
    alpha=1,
    lw=1,
    precision=3,
    label_rotation=0,
    axis_rotation=0,
    tick_angle=90,
) -> Mesh:
    """
    Build a 3D ruler to indicate the distance of two points p1 and p2.

    Arguments:
        label : (str)
            alternative fixed label to be shown
        units_scale : (float)
            factor to scale units (e.g. μm to mm)
        s : (float)
            size of the label
        font : (str)
            font face.  Check [available fonts here](https://vedo.embl.es/fonts).
        italic : (float)
            italicness of the font in the range [0,1]
        units : (str)
            string to be appended to the numeric value
        lw : (int)
            line width in pixel units
        precision : (int)
            nr of significant digits to be shown
        label_rotation : (float)
            initial rotation of the label around the z-axis
        axis_rotation : (float)
            initial rotation of the line around the main axis
        tick_angle : (float)
            initial rotation of the line around the main axis

    Examples:
        - [goniometer.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/goniometer.py)

        ![](https://vedo.embl.es/images/pyplot/goniometer.png)
    """

    if units_scale != 1.0 and units == "":
        raise ValueError(
            "When setting 'units_scale' to a value other than 1, "
            + "a 'units' arguments must be specified."
        )

    try:
        p1 = p1.pos()
    except AttributeError:
        pass

    try:
        p2 = p2.pos()
    except AttributeError:
        pass

    if len(p1) == 2:
        p1 = [p1[0], p1[1], 0.0]
    if len(p2) == 2:
        p2 = [p2[0], p2[1], 0.0]

    p1, p2 = np.asarray(p1), np.asarray(p2)
    q1, q2 = [0, 0, 0], [utils.mag(p2 - p1), 0, 0]
    q1, q2 = np.array(q1), np.array(q2)
    v = q2 - q1
    d = utils.mag(v) * units_scale

    pos = np.array(p1)
    p1 = p1 - pos
    p2 = p2 - pos

    if s is None:
        s = d * 0.02 * (1 / units_scale)

    if not label:
        label = str(d)
        if precision:
            label = utils.precision(d, precision)
    if prefix:
        label = prefix + "~" + label
    if units:
        label += "~" + units

    lb = shapes.Text3D(label, s=s, font=font, italic=italic, justify="center")
    if label_rotation:
        lb.rotate_z(label_rotation)
    lb.pos((q1 + q2) / 2)

    x0, x1 = lb.xbounds()
    gap = [(x1 - x0) / 2, 0, 0]
    pc1 = (v / 2 - gap) * 0.9 + q1
    pc2 = q2 - (v / 2 - gap) * 0.9

    lc1 = shapes.Line(q1 - v / 50, pc1).lw(lw)
    lc2 = shapes.Line(q2 + v / 50, pc2).lw(lw)

    zs = np.array([0, d / 50 * (1 / units_scale), 0])
    ml1 = shapes.Line(-zs, zs).lw(lw)
    ml2 = shapes.Line(-zs, zs).lw(lw)
    ml1.rotate_z(tick_angle - 90).pos(q1)
    ml2.rotate_z(tick_angle - 90).pos(q2)

    c1 = shapes.Circle(q1, r=d / 180 * (1 / units_scale), res=24)
    c2 = shapes.Circle(q2, r=d / 180 * (1 / units_scale), res=24)

    macts = merge(lb, lc1, lc2, c1, c2, ml1, ml2)
    macts.c(c).alpha(alpha)
    macts.properties.SetLineWidth(lw)
    macts.properties.LightingOff()
    macts.actor.UseBoundsOff()
    macts.rotate_x(axis_rotation)
    macts.reorient(q2 - q1, p2 - p1)
    macts.pos(pos)
    macts.bc("tomato").pickable(False)
    return macts


def RulerAxes(
    inputobj,
    xtitle="",
    ytitle="",
    ztitle="",
    xlabel="",
    ylabel="",
    zlabel="",
    xpadding=0.05,
    ypadding=0.04,
    zpadding=0,
    font="Normografo",
    s=None,
    italic=0,
    units="",
    c=(0.2, 0, 0),
    alpha=1,
    lw=1,
    precision=3,
    label_rotation=0,
    xaxis_rotation=0,
    yaxis_rotation=0,
    zaxis_rotation=0,
    xycross=True,
) -> Mesh | None:
    """
    A 3D ruler axes to indicate the sizes of the input scene or object.

    Arguments:
        xtitle : (str)
            name of the axis or title
        xlabel : (str)
            alternative fixed label to be shown instead of the distance
        s : (float)
            size of the label
        font : (str)
            font face. Check [available fonts here](https://vedo.embl.es/fonts).
        italic : (float)
            italicness of the font in the range [0,1]
        units : (str)
            string to be appended to the numeric value
        lw : (int)
            line width in pixel units
        precision : (int)
            nr of significant digits to be shown
        label_rotation : (float)
            initial rotation of the label around the z-axis
        [x,y,z]axis_rotation : (float)
            initial rotation of the line around the main axis in degrees
        xycross : (bool)
            show two back crossing lines in the xy plane

    Examples:
        - [goniometer.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/goniometer.py)
    """
    if utils.is_sequence(inputobj):
        x0, x1, y0, y1, z0, z1 = inputobj
    else:
        x0, x1, y0, y1, z0, z1 = inputobj.bounds()
    dx, dy, dz = (y1 - y0) * xpadding, (x1 - x0) * ypadding, (y1 - y0) * zpadding
    d = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2 + (z1 - z0) ** 2)

    if not d:
        return None

    if s is None:
        s = d / 75

    acts, rx, ry = [], None, None
    if xtitle is not None and (x1 - x0) / d > 0.1:
        rx = Ruler3D(
            [x0, y0 - dx, z0],
            [x1, y0 - dx, z0],
            s=s,
            font=font,
            precision=precision,
            label_rotation=label_rotation,
            axis_rotation=xaxis_rotation,
            lw=lw,
            italic=italic,
            prefix=xtitle,
            label=xlabel,
            units=units,
        )
        acts.append(rx)

    if ytitle is not None and (y1 - y0) / d > 0.1:
        ry = Ruler3D(
            [x1 + dy, y0, z0],
            [x1 + dy, y1, z0],
            s=s,
            font=font,
            precision=precision,
            label_rotation=label_rotation,
            axis_rotation=yaxis_rotation,
            lw=lw,
            italic=italic,
            prefix=ytitle,
            label=ylabel,
            units=units,
        )
        acts.append(ry)

    if ztitle is not None and (z1 - z0) / d > 0.1:
        rz = Ruler3D(
            [x0 - dy, y0 + dz, z0],
            [x0 - dy, y0 + dz, z1],
            s=s,
            font=font,
            precision=precision,
            label_rotation=label_rotation,
            axis_rotation=zaxis_rotation + 90,
            lw=lw,
            italic=italic,
            prefix=ztitle,
            label=zlabel,
            units=units,
        )
        acts.append(rz)

    if xycross and rx and ry:
        lx = shapes.Line([x0, y0, z0], [x0, y1 + dx, z0])
        ly = shapes.Line([x0 - dy, y1, z0], [x1, y1, z0])
        d = min((x1 - x0), (y1 - y0)) / 200
        cxy = shapes.Circle([x0, y1, z0], r=d, res=15)
        acts.extend([lx, ly, cxy])

    macts = merge(acts)
    if not macts:
        return None
    macts.c(c).alpha(alpha).bc("t")
    macts.actor.UseBoundsOff()
    macts.actor.PickableOff()
    return macts


#####################################################################
