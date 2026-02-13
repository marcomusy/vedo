#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Advanced curve connectors and arrow-like helpers."""

from typing import Union
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import utils
from vedo.transformations import LinearTransform
from vedo.colors import get_color
from vedo.mesh import Mesh
from vedo.shapes.glyphs import Glyph
from vedo.shapes.curves_core import Line

class Ribbon(Mesh):
    """
    Connect two lines to generate the surface inbetween.
    Set the mode by which to create the ruled surface.

    It also works with a single line in input. In this case the ribbon
    is formed by following the local plane of the line in space.
    """

    def __init__(
        self,
        line1,
        line2=None,
        mode=0,
        closed=False,
        width=None,
        res=(200, 5),
        c="indigo3",
        alpha=1.0,
    ) -> None:
        """
        Arguments:
            mode : (int)
                If mode=0, resample evenly the input lines (based on length)
                and generates triangle strips.

                If mode=1, use the existing points and walks around the
                polyline using existing points.

            closed : (bool)
                if True, join the last point with the first to form a closed surface

            res : (list)
                ribbon resolutions along the line and perpendicularly to it.

        Examples:
            - [ribbon.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/ribbon.py)

                ![](https://vedo.embl.es/images/basic/ribbon.png)
        """

        if isinstance(line1, Points):
            line1 = line1.coordinates

        if isinstance(line2, Points):
            line2 = line2.coordinates

        elif line2 is None:
            #############################################
            ribbon_filter = vtki.new("RibbonFilter")
            aline = Line(line1)
            ribbon_filter.SetInputData(aline.dataset)
            if width is None:
                width = aline.diagonal_size() / 20.0
            ribbon_filter.SetWidth(width)
            ribbon_filter.Update()
            # convert triangle strips to polygons
            tris = vtki.new("TriangleFilter")
            tris.SetInputData(ribbon_filter.GetOutput())
            tris.Update()

            super().__init__(tris.GetOutput(), c, alpha)
            self.name = "Ribbon"
            ##############################################
            return  ######################################
            ##############################################

        line1 = np.asarray(line1)
        line2 = np.asarray(line2)

        if closed:
            line1 = line1.tolist()
            line1 += [line1[0]]
            line2 = line2.tolist()
            line2 += [line2[0]]
            line1 = np.array(line1)
            line2 = np.array(line2)

        if len(line1[0]) == 2:
            line1 = np.c_[line1, np.zeros(len(line1))]
        if len(line2[0]) == 2:
            line2 = np.c_[line2, np.zeros(len(line2))]

        ppoints1 = vtki.vtkPoints()  # Generate the polyline1
        ppoints1.SetData(utils.numpy2vtk(line1, dtype=np.float32))
        lines1 = vtki.vtkCellArray()
        lines1.InsertNextCell(len(line1))
        for i in range(len(line1)):
            lines1.InsertCellPoint(i)
        poly1 = vtki.vtkPolyData()
        poly1.SetPoints(ppoints1)
        poly1.SetLines(lines1)

        ppoints2 = vtki.vtkPoints()  # Generate the polyline2
        ppoints2.SetData(utils.numpy2vtk(line2, dtype=np.float32))
        lines2 = vtki.vtkCellArray()
        lines2.InsertNextCell(len(line2))
        for i in range(len(line2)):
            lines2.InsertCellPoint(i)
        poly2 = vtki.vtkPolyData()
        poly2.SetPoints(ppoints2)
        poly2.SetLines(lines2)

        # build the lines
        lines1 = vtki.vtkCellArray()
        lines1.InsertNextCell(poly1.GetNumberOfPoints())
        for i in range(poly1.GetNumberOfPoints()):
            lines1.InsertCellPoint(i)

        polygon1 = vtki.vtkPolyData()
        polygon1.SetPoints(ppoints1)
        polygon1.SetLines(lines1)

        lines2 = vtki.vtkCellArray()
        lines2.InsertNextCell(poly2.GetNumberOfPoints())
        for i in range(poly2.GetNumberOfPoints()):
            lines2.InsertCellPoint(i)

        polygon2 = vtki.vtkPolyData()
        polygon2.SetPoints(ppoints2)
        polygon2.SetLines(lines2)

        merged_pd = vtki.new("AppendPolyData")
        merged_pd.AddInputData(polygon1)
        merged_pd.AddInputData(polygon2)
        merged_pd.Update()

        rsf = vtki.new("RuledSurfaceFilter")
        rsf.CloseSurfaceOff()
        rsf.SetRuledMode(mode)
        rsf.SetResolution(res[0], res[1])
        rsf.SetInputData(merged_pd.GetOutput())
        rsf.Update()
        # convert triangle strips to polygons
        tris = vtki.new("TriangleFilter")
        tris.SetInputData(rsf.GetOutput())
        tris.Update()
        out = tris.GetOutput()

        super().__init__(out, c, alpha)

        self.name = "Ribbon"


class Arrow(Mesh):
    """
    Build a 3D arrow from `start_pt` to `end_pt` of section size `s`,
    expressed as the fraction of the window size.
    """

    def __init__(
        self,
        start_pt=(0, 0, 0),
        end_pt=(1, 0, 0),
        s=None,
        shaft_radius=None,
        head_radius=None,
        head_length=None,
        res=12,
        c="r4",
        alpha=1.0,
    ) -> None:
        """
        If `c` is a `float` less than 1, the arrow is rendered as a in a color scale
        from white to red.

        .. note:: If `s=None` the arrow is scaled proportionally to its length

        ![](https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestOrientedArrow.png)
        """
        # in case user is passing meshs
        if isinstance(start_pt, vtki.vtkActor):
            start_pt = start_pt.GetPosition()
        if isinstance(end_pt, vtki.vtkActor):
            end_pt = end_pt.GetPosition()

        axis = np.asarray(end_pt) - np.asarray(start_pt)
        length = float(np.linalg.norm(axis))
        if length:
            axis = axis / length
        if len(axis) < 3:  # its 2d
            theta = np.pi / 2
            start_pt = [start_pt[0], start_pt[1], 0.0]
            end_pt = [end_pt[0], end_pt[1], 0.0]
        else:
            theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])
        self.source = vtki.new("ArrowSource")
        self.source.SetShaftResolution(res)
        self.source.SetTipResolution(res)

        if s:
            sz = 0.02
            self.source.SetTipRadius(sz)
            self.source.SetShaftRadius(sz / 1.75)
            self.source.SetTipLength(sz * 15)

        if head_length:
            self.source.SetTipLength(head_length)
        if head_radius:
            self.source.SetTipRadius(head_radius)
        if shaft_radius:
            self.source.SetShaftRadius(shaft_radius)

        self.source.Update()

        t = vtki.vtkTransform()
        t.Translate(start_pt)
        t.RotateZ(np.rad2deg(phi))
        t.RotateY(np.rad2deg(theta))
        t.RotateY(-90)  # put it along Z
        if s:
            sz = 800 * s
            t.Scale(length, sz, sz)
        else:
            t.Scale(length, length, length)

        tf = vtki.new("TransformPolyDataFilter")
        tf.SetInputData(self.source.GetOutput())
        tf.SetTransform(t)
        tf.Update()

        super().__init__(tf.GetOutput(), c, alpha)

        self.transform = LinearTransform().translate(start_pt)

        self.phong().lighting("plastic")
        self.actor.PickableOff()
        self.actor.DragableOff()
        self.base = np.array(start_pt, dtype=float)  # used by pyplot
        self.top  = np.array(end_pt,   dtype=float)  # used by pyplot
        self.top_index = self.source.GetTipResolution() * 4
        self.fill = True                    # used by pyplot.__iadd__()
        self.s = s if s is not None else 1  # used by pyplot.__iadd__()
        self.name = "Arrow"

    def top_point(self):
        """Return the current coordinates of the tip of the Arrow."""
        return self.transform.transform_point(self.top)

    def base_point(self):
        """Return the current coordinates of the base of the Arrow."""
        return self.transform.transform_point(self.base)

class Arrows(Glyph):
    """
    Build arrows between two lists of points.
    """

    def __init__(
        self,
        start_pts,
        end_pts=None,
        s=None,
        shaft_radius=None,
        head_radius=None,
        head_length=None,
        thickness=1.0,
        res=6,
        c='k3',
        alpha=1.0,
    ) -> None:
        """
        Build arrows between two lists of points `start_pts` and `end_pts`.
         `start_pts` can be also passed in the form `[[point1, point2], ...]`.

        Color can be specified as a colormap which maps the size of the arrows.

        Arguments:
            s : (float)
                fix aspect-ratio of the arrow and scale its cross section
            c : (color)
                color or color map name
            alpha : (float)
                set object opacity
            res : (int)
                set arrow resolution

        Examples:
            - [glyphs2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/glyphs2.py)

            ![](https://user-images.githubusercontent.com/32848391/55897850-a1a0da80-5bc1-11e9-81e0-004c8f396b43.jpg)
        """
        if isinstance(start_pts, Points):
            start_pts = start_pts.coordinates
        if isinstance(end_pts, Points):
            end_pts = end_pts.coordinates

        start_pts = np.asarray(start_pts)
        if end_pts is None:
            strt = start_pts[:, 0]
            end_pts = start_pts[:, 1]
            start_pts = strt
        else:
            end_pts = np.asarray(end_pts)

        start_pts = utils.make3d(start_pts)
        end_pts = utils.make3d(end_pts)

        arr = vtki.new("ArrowSource")
        arr.SetShaftResolution(res)
        arr.SetTipResolution(res)

        if s:
            sz = 0.02 * s
            arr.SetTipRadius(sz * 2)
            arr.SetShaftRadius(sz * thickness)
            arr.SetTipLength(sz * 10)

        if head_radius:
            arr.SetTipRadius(head_radius)
        if shaft_radius:
            arr.SetShaftRadius(shaft_radius)
        if head_length:
            arr.SetTipLength(head_length)

        arr.Update()
        out = arr.GetOutput()

        orients = end_pts - start_pts

        color_by_vector_size = utils.is_sequence(c) or c in cmaps_names

        super().__init__(
            start_pts,
            out,
            orientation_array=orients,
            scale_by_vector_size=True,
            color_by_vector_size=color_by_vector_size,
            c=c,
            alpha=alpha,
        )
        self.lighting("off")
        self.actor.PickableOff()
        self.actor.DragableOff()
        if color_by_vector_size:
            vals = np.linalg.norm(orients, axis=1)
            self.mapper.SetScalarRange(vals.min(), vals.max())
        else:
            self.c(c)
        self.name = "Arrows"


class Arrow2D(Mesh):
    """
    Build a 2D arrow.
    """

    def __init__(
        self,
        start_pt=(0, 0, 0),
        end_pt=(1, 0, 0),
        s=1,
        rotation=0.0,
        shaft_length=0.85,
        shaft_width=0.055,
        head_length=0.175,
        head_width=0.175,
        fill=True,
        c="red4",
        alpha=1.0,
   ) -> None:
        """
        Build a 2D arrow from `start_pt` to `end_pt`.

        Arguments:
            s : (float)
                a global multiplicative convenience factor controlling the arrow size
            shaft_length : (float)
                fractional shaft length
            shaft_width : (float)
                fractional shaft width
            head_length : (float)
                fractional head length
            head_width : (float)
                fractional head width
            fill : (bool)
                if False only generate the outline
        """
        self.fill = fill  ## needed by pyplot.__iadd()
        self.s = s        ## needed by pyplot.__iadd()

        if s != 1:
            shaft_width *= s
            head_width *= np.sqrt(s)

        # in case user is passing meshs
        if isinstance(start_pt, vtki.vtkActor):
            start_pt = start_pt.GetPosition()
        if isinstance(end_pt, vtki.vtkActor):
            end_pt = end_pt.GetPosition()
        if len(start_pt) == 2:
            start_pt = [start_pt[0], start_pt[1], 0]
        if len(end_pt) == 2:
            end_pt = [end_pt[0], end_pt[1], 0]

        headBase = 1 - head_length
        head_width = max(head_width, shaft_width)
        if head_length is None or headBase > shaft_length:
            headBase = shaft_length

        verts = []
        verts.append([0, -shaft_width / 2, 0])
        verts.append([shaft_length, -shaft_width / 2, 0])
        verts.append([headBase, -head_width / 2, 0])
        verts.append([1, 0, 0])
        verts.append([headBase, head_width / 2, 0])
        verts.append([shaft_length, shaft_width / 2, 0])
        verts.append([0, shaft_width / 2, 0])
        if fill:
            faces = ((0, 1, 3, 5, 6), (5, 3, 4), (1, 2, 3))
            poly = utils.buildPolyData(verts, faces)
        else:
            lines = (0, 1, 2, 3, 4, 5, 6, 0)
            poly = utils.buildPolyData(verts, [], lines=lines)

        axis = np.array(end_pt) - np.array(start_pt)
        length = float(np.linalg.norm(axis))
        if length:
            axis = axis / length
        theta = 0
        if len(axis) > 2:
            theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])

        t = vtki.vtkTransform()
        t.Translate(start_pt)
        if phi:
            t.RotateZ(np.rad2deg(phi))
        if theta:
            t.RotateY(np.rad2deg(theta))
        t.RotateY(-90)  # put it along Z
        if rotation:
            t.RotateX(rotation)
        t.Scale(length, length, length)

        tf = vtki.new("TransformPolyDataFilter")
        tf.SetInputData(poly)
        tf.SetTransform(t)
        tf.Update()

        super().__init__(tf.GetOutput(), c, alpha)

        self.transform = LinearTransform().translate(start_pt)

        self.lighting("off")
        self.actor.DragableOff()
        self.actor.PickableOff()
        self.base = np.array(start_pt, dtype=float) # used by pyplot
        self.top  = np.array(end_pt,   dtype=float) # used by pyplot
        self.name = "Arrow2D"


class Arrows2D(Glyph):
    """
    Build 2D arrows between two lists of points.
    """

    def __init__(
        self,
        start_pts,
        end_pts=None,
        s=1.0,
        rotation=0.0,
        shaft_length=0.8,
        shaft_width=0.05,
        head_length=0.225,
        head_width=0.175,
        fill=True,
        c=None,
        alpha=1.0,
    ) -> None:
        """
        Build 2D arrows between two lists of points `start_pts` and `end_pts`.
        `start_pts` can be also passed in the form `[[point1, point2], ...]`.

        Color can be specified as a colormap which maps the size of the arrows.

        Arguments:
            shaft_length : (float)
                fractional shaft length
            shaft_width : (float)
                fractional shaft width
            head_length : (float)
                fractional head length
            head_width : (float)
                fractional head width
            fill : (bool)
                if False only generate the outline
        """
        if isinstance(start_pts, Points):
            start_pts = start_pts.coordinates
        if isinstance(end_pts, Points):
            end_pts = end_pts.coordinates

        start_pts = np.asarray(start_pts, dtype=float)
        if end_pts is None:
            strt = start_pts[:, 0]
            end_pts = start_pts[:, 1]
            start_pts = strt
        else:
            end_pts = np.asarray(end_pts, dtype=float)

        if head_length is None:
            head_length = 1 - shaft_length

        arr = Arrow2D(
            (0, 0, 0),
            (1, 0, 0),
            s=s,
            rotation=rotation,
            shaft_length=shaft_length,
            shaft_width=shaft_width,
            head_length=head_length,
            head_width=head_width,
            fill=fill,
        )

        orients = end_pts - start_pts
        orients = utils.make3d(orients)

        pts = Points(start_pts)
        super().__init__(
            pts,
            arr,
            orientation_array=orients,
            scale_by_vector_size=True,
            c=c,
            alpha=alpha,
        )
        self.flat().lighting("off")
        self.actor.PickableOff()
        self.actor.DragableOff()
        if c is not None:
            self.color(c)
        self.name = "Arrows2D"


class FlatArrow(Ribbon):
    """
    Build a 2D arrow in 3D space by joining two close lines.
    """

    def __init__(self, line1, line2, tip_size=1.0, tip_width=1.0) -> None:
        """
        Build a 2D arrow in 3D space by joining two close lines.

        Examples:
            - [flatarrow.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/flatarrow.py)

                ![](https://vedo.embl.es/images/basic/flatarrow.png)
        """
        if isinstance(line1, Points):
            line1 = line1.coordinates
        if isinstance(line2, Points):
            line2 = line2.coordinates

        sm1, sm2 = np.array(line1[-1], dtype=float), np.array(line2[-1], dtype=float)

        v = (sm1 - sm2) / 3 * tip_width
        p1 = sm1 + v
        p2 = sm2 - v
        pm1 = (sm1 + sm2) / 2
        pm2 = (np.array(line1[-2]) + np.array(line2[-2])) / 2
        pm12 = pm1 - pm2
        tip = pm12 / np.linalg.norm(pm12) * np.linalg.norm(v) * 3 * tip_size / tip_width + pm1

        line1.append(p1)
        line1.append(tip)
        line2.append(p2)
        line2.append(tip)
        resm = max(100, len(line1))

        super().__init__(line1, line2, res=(resm, 1))
        self.phong().lighting("off")
        self.actor.PickableOff()
        self.actor.DragableOff()
        self.name = "FlatArrow"


