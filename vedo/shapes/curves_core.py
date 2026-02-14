#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Core curve and spline primitives."""

from typing import Any
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import utils
from vedo.transformations import LinearTransform, pol2cart, cart2spher, spher2cart
from vedo.colors import get_color, printc
from vedo.pointcloud import Points, merge
from vedo.mesh import Mesh

class Line(Mesh):
    """
    Build the line segment between point `p0` and point `p1`.

    If `p0` is already a list of points, return the line connecting them.

    A 2D set of coords can also be passed as `p0=[x..], p1=[y..]`.
    """

    def __init__(self, p0, p1=None, closed=False, res=2, lw=1, c="k1", alpha=1.0) -> None:
        """
        Arguments:
            closed : (bool)
                join last to first point
            res : (int)
                resolution, number of points along the line
                (only relevant if only 2 points are specified)
            lw : (int)
                line width in pixel units
        """

        if isinstance(p1, Points):
            p1 = p1.pos()
            if isinstance(p0, Points):
                p0 = p0.pos()
        try:
            p0 = p0.dataset
        except AttributeError:
            pass

        if isinstance(p0, vtki.vtkPolyData):
            poly = p0
            top  = np.array([0,0,1])
            base = np.array([0,0,0])

        elif utils.is_sequence(p0[0]): # detect if user is passing a list of points

            p0 = utils.make3d(p0)
            ppoints = vtki.vtkPoints()  # Generate the polyline
            ppoints.SetData(utils.numpy2vtk(np.asarray(p0), dtype=np.float32))
            lines = vtki.vtkCellArray()
            npt = len(p0)
            if closed:
                lines.InsertNextCell(npt + 1)
            else:
                lines.InsertNextCell(npt)
            for i in range(npt):
                lines.InsertCellPoint(i)
            if closed:
                lines.InsertCellPoint(0)
            poly = vtki.vtkPolyData()
            poly.SetPoints(ppoints)
            poly.SetLines(lines)
            top = p0[-1]
            base = p0[0]
            if res != 2:
                printc(f"Warning: calling Line(res={res}), try remove []?", c='y')
                res = 2

        else:  # or just 2 points to link

            line_source = vtki.new("LineSource")
            p0 = utils.make3d(p0)
            p1 = utils.make3d(p1)
            line_source.SetPoint1(p0)
            line_source.SetPoint2(p1)
            line_source.SetResolution(res - 1)
            line_source.Update()
            poly = line_source.GetOutput()
            top = np.asarray(p1, dtype=float)
            base = np.asarray(p0, dtype=float)

        super().__init__(poly, c, alpha)

        self.slope: list[float] = []  # populated by analysis.fit_line
        self.center: list[float] = []
        self.variances: list[float] = []

        self.coefficients: list[float] = []  # populated by pyplot.fit()
        self.covariance_matrix: list[float] = []
        self.coefficient_errors: list[float] = []
        self.monte_carlo_coefficients: list[float] = []
        self.reduced_chi2 = -1
        self.ndof = 0
        self.data_sigma = 0
        self.error_lines: list[Any] = []
        self.error_band = None
        self.res = res
        self.is_closed = closed

        self.lw(lw)
        self.properties.LightingOff()
        self.actor.PickableOff()
        self.actor.DragableOff()
        self.base = base
        self.top = top
        self.name = "Line"

    def clone(self, deep=True) -> "Line":
        """
        Return a copy of the ``Line`` object.

        Example:
            ```python
            from vedo import *
            ln1 = Line([1,1,1], [2,2,2], lw=3).print()
            ln2 = ln1.clone().shift(0,0,1).c('red').print()
            show(ln1, ln2, axes=1, viewup='z').close()
            ```
            ![](https://vedo.embl.es/images/feats/line_clone.png)
        """
        poly = vtki.vtkPolyData()
        if deep:
            poly.DeepCopy(self.dataset)
        else:
            poly.ShallowCopy(self.dataset)
        ln = Line(poly)
        ln.copy_properties_from(self)
        ln.transform = self.transform.clone()
        ln.name = self.name
        ln.base = self.base
        ln.top = self.top
        ln.pipeline = utils.OperationNode(
            "clone", parents=[self], shape="diamond", c="#edede9")
        return ln

    def linecolor(self, lc=None) -> "Line":
        """Assign a color to the line"""
        # overrides mesh.linecolor which would have no effect here
        return self.color(lc)

    def eval(self, x: float) -> np.ndarray:
        """
        Calculate the position of an intermediate point
        as a fraction of the length of the line,
        being x=0 the first point and x=1 the last point.
        This corresponds to an imaginary point that travels along the line
        at constant speed.

        Can be used in conjunction with `lin_interpolate()`
        to map any range to the [0,1] range.
        """
        distance1 = 0.0
        length = self.length()
        pts = self.coordinates
        if self.is_closed:
            pts = np.append(pts, [pts[0]], axis=0)

        for i in range(1, len(pts)):
            p0 = pts[i - 1]
            p1 = pts[i]
            seg = p1 - p0
            distance0 = distance1
            distance1 += np.linalg.norm(seg)
            w1 = distance1 / length
            if w1 >= x:
                break
        w0 = distance0 / length
        v = p0 + seg * (x - w0) / (w1 - w0)
        return v

    def eval2d(self, x: float) -> np.ndarray:
        """
        Calculate the position of an intermediate point
        at the specified value of x in absolute units.
        Assume the line is in the xy-plane.
        """
        xcoords, ycoords, _ = self.coordinates.T
        # find the segment where x is located
        idx = np.where((xcoords[:-1] <= x) & (xcoords[1:] >= x))[0]
        if len(idx) > 0:
            i = idx[0]
            return np.array([x, np.interp(x, xcoords[i:i+2], ycoords[i:i+2])])
        return np.array([x, 0.0])

    def find_index_at_position(self, p) -> float:
        """
        Find the index of the line vertex that is closest to the point `p`.
        Note that the returned index is fractional as `p` may not be exactly
        one of the vertices of the line.
        """
        tf = vtki.new("TriangleFilter")
        tf.SetPassLines(True)
        tf.SetPassVerts(False)
        tf.SetInputData(self.dataset)
        tf.Update()
        polyline = tf.GetOutput()

        if not self.cell_locator:
            self.cell_locator = vtki.new("StaticCellLocator")
            self.cell_locator.SetDataSet(polyline)
            self.cell_locator.BuildLocator()
        
        q = [0, 0, 0]
        cid = vtki.mutable(0)
        dist2 = vtki.mutable(0)
        subid = vtki.mutable(0)
        self.cell_locator.FindClosestPoint(p, q, cid, subid, dist2)

        # find the 2 points
        a = polyline.GetCell(cid).GetPointId(0)
        b = polyline.GetCell(cid).GetPointId(1)

        pts = self.coordinates
        if self.is_closed:
            pts = np.append(pts, [pts[0]], axis=0)
        d = np.linalg.norm(pts[a] - pts[b])
        t = a + np.linalg.norm(pts[a] - q) / d
        return t

    def pattern(self, stipple, repeats=10) -> "Line":
        """
        Define a stipple pattern for dashing the line.
        Pass the stipple pattern as a string like `'- - -'`.
        Repeats controls the number of times the pattern repeats in a single segment.

        Examples are: `'- -', '--  -  --'`, etc.

        The resolution of the line (nr of points) can affect how pattern will show up.

        Example:
            ```python
            from vedo import Line
            pts = [[1, 0, 0], [5, 2, 0], [3, 3, 1]]
            ln = Line(pts, c='r', lw=5).pattern('- -', repeats=10)
            ln.show(axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/line_pattern.png)
        """
        stipple = str(stipple) * int(2 * repeats)
        dimension = len(stipple)

        image = vtki.vtkImageData()
        image.SetDimensions(dimension, 1, 1)
        image.AllocateScalars(vtki.VTK_UNSIGNED_CHAR, 4)
        image.SetExtent(0, dimension - 1, 0, 0, 0, 0)
        i_dim = 0
        while i_dim < dimension:
            for i in range(dimension):
                image.SetScalarComponentFromFloat(i_dim, 0, 0, 0, 255)
                image.SetScalarComponentFromFloat(i_dim, 0, 0, 1, 255)
                image.SetScalarComponentFromFloat(i_dim, 0, 0, 2, 255)
                if stipple[i] == " ":
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 3, 0)
                else:
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 3, 255)
                i_dim += 1

        poly = self.dataset

        # Create texture coordinates
        tcoords = vtki.vtkDoubleArray()
        tcoords.SetName("TCoordsStippledLine")
        tcoords.SetNumberOfComponents(1)
        tcoords.SetNumberOfTuples(poly.GetNumberOfPoints())
        for i in range(poly.GetNumberOfPoints()):
            tcoords.SetTypedTuple(i, [i / 2])
        poly.GetPointData().SetTCoords(tcoords)
        poly.GetPointData().Modified()
        texture = vtki.vtkTexture()
        texture.SetInputData(image)
        texture.InterpolateOff()
        texture.RepeatOn()
        self.actor.SetTexture(texture)
        return self

    def length(self) -> float:
        """Calculate length of the line."""
        pts = self.coordinates
        if self.is_closed:
            pts = np.append(pts, [pts[0]], axis=0)
        distance = 0.0
        for i in range(1, len(pts)):
            distance += np.linalg.norm(pts[i] - pts[i - 1])
        return distance

    def tangents(self) -> np.ndarray:
        """
        Compute the tangents of a line in space.

        Example:
            ```python
            from vedo import *
            shape = Assembly(dataurl+"timecourse1d.npy")[58]
            pts = shape.rotate_x(30).coordinates
            tangents = Line(pts).tangents()
            arrs = Arrows(pts, pts+tangents, c='blue9')
            show(shape.c('red5').lw(5), arrs, bg='bb', axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/line_tangents.png)
        """
        v = np.gradient(self.coordinates)[0]
        ds_dt = np.linalg.norm(v, axis=1)
        tangent = np.array([1 / ds_dt] * 3).transpose() * v
        return tangent

    def curvature(self) -> np.ndarray:
        """
        Compute the signed curvature of a line in space.
        The signed is computed assuming the line is about coplanar to the xy plane.

        Example:
            ```python
            from vedo import *
            from vedo.pyplot import plot
            shape = Assembly(dataurl+"timecourse1d.npy")[55]
            curvs = Line(shape.coordinates).curvature()
            shape.cmap('coolwarm', curvs, vmin=-2,vmax=2).add_scalarbar3d(c='w')
            shape.render_lines_as_tubes().lw(12)
            pp = plot(curvs, ac='white', lc='yellow5')
            show(shape, pp, N=2, bg='bb', sharecam=False).close()
            ```
            ![](https://vedo.embl.es/images/feats/line_curvature.png)
        """
        v = np.gradient(self.coordinates)[0]
        a = np.gradient(v)[0]
        av = np.cross(a, v)
        mav = np.linalg.norm(av, axis=1)
        mv = utils.mag2(v)
        val = mav * np.sign(av[:, 2]) / np.power(mv, 1.5)
        val[0] = val[1]
        val[-1] = val[-2]
        return val

    def compute_curvature(self, method=0) -> "Line":
        """
        Add a pointdata array named 'Curvatures' which contains
        the curvature value at each point.

        NB: keyword `method` is overridden in Mesh and has no effect here.
        """
        # overrides mesh.compute_curvature
        curvs = self.curvature()
        vmin, vmax = np.min(curvs), np.max(curvs)
        if vmin < 0 and vmax > 0:
            v = max(-vmin, vmax)
            self.cmap("coolwarm", curvs, vmin=-v, vmax=v, name="Curvature")
        else:
            self.cmap("coolwarm", curvs, vmin=vmin, vmax=vmax, name="Curvature")
        return self

    def plot_scalar(
            self,
            radius=0.0,
            height=1.1,
            normal=(),
            camera=None,
        ) -> "Line":
        """
        Generate a new `Line` which plots the active scalar along the line.

        Arguments:
            radius : (float)
                distance radius to the line
            height: (float)
                height of the plot
            normal: (list)
                normal vector to the plane of the plot
            camera: (vtkCamera)
                camera object to use for the plot orientation

        Example:
            ```python
            from vedo import *
            circle = Circle(res=360).rotate_y(20)
            pts = circle.coordinates
            bore = Line(pts).lw(5)
            values = np.arctan2(pts[:,1], pts[:,0])
            bore.pointdata["scalars"] = values + np.random.randn(360)/5
            vap = bore.plot_scalar(radius=0, height=1)
            show(bore, vap, axes=1, viewup='z').close()
            ```
            ![](https://vedo.embl.es/images/feats/line_plot_scalar.png)
        """
        ap = vtki.new("ArcPlotter")
        ap.SetInputData(self.dataset)
        ap.SetCamera(camera)
        ap.SetRadius(radius)
        ap.SetHeight(height)
        if len(normal)>0:
            ap.UseDefaultNormalOn()
            ap.SetDefaultNormal(normal)
        ap.Update()
        vap = Line(ap.GetOutput())
        vap.linewidth(3).lighting('off')
        vap.name = "ArcPlot"
        return vap

    def sweep(self, direction=(1, 0, 0), res=1) -> "Mesh":
        """
        Sweep the `Line` along the specified vector direction.

        Returns a `Mesh` surface.
        Line position is updated to allow for additional sweepings.

        Example:
            ```python
            from vedo import Line, show
            aline = Line([(0,0,0),(1,3,0),(2,4,0)])
            surf1 = aline.sweep((1,0.2,0), res=3)
            surf2 = aline.sweep((0.2,0,1)).alpha(0.5)
            aline.color('r').linewidth(4)
            show(surf1, surf2, aline, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/sweepline.png)
        """
        line = self.dataset
        rows = line.GetNumberOfPoints()

        spacing = 1 / res
        surface = vtki.vtkPolyData()

        res += 1
        npts = rows * res
        npolys = (rows - 1) * (res - 1)
        points = vtki.vtkPoints()
        points.Allocate(npts)

        cnt = 0
        x = [0.0, 0.0, 0.0]
        for row in range(rows):
            for col in range(res):
                p = [0.0, 0.0, 0.0]
                line.GetPoint(row, p)
                x[0] = p[0] + direction[0] * col * spacing
                x[1] = p[1] + direction[1] * col * spacing
                x[2] = p[2] + direction[2] * col * spacing
                points.InsertPoint(cnt, x)
                cnt += 1

        # Generate the quads
        polys = vtki.vtkCellArray()
        polys.Allocate(npolys * 4)
        pts = [0, 0, 0, 0]
        for row in range(rows - 1):
            for col in range(res - 1):
                pts[0] = col + row * res
                pts[1] = pts[0] + 1
                pts[2] = pts[0] + res + 1
                pts[3] = pts[0] + res
                polys.InsertNextCell(4, pts)
        surface.SetPoints(points)
        surface.SetPolys(polys)
        asurface = Mesh(surface)
        asurface.copy_properties_from(self)
        asurface.lighting("default")
        self.coordinates = self.coordinates + direction
        return asurface

    def reverse(self):
        """Reverse the points sequence order."""
        pts = np.flip(self.coordinates, axis=0)
        self.coordinates = pts
        return self


class DashedLine(Mesh):
    """
    Consider using `Line.pattern()` instead.

    Build a dashed line segment between points `p0` and `p1`.
    If `p0` is a list of points returns the line connecting them.
    A 2D set of coords can also be passed as `p0=[x..], p1=[y..]`.
    """

    def __init__(self, p0, p1=None, spacing=0.1, closed=False, lw=2, c="k5", alpha=1.0) -> None:
        """
        Arguments:
            closed : (bool)
                join last to first point
            spacing : (float)
                relative size of the dash
            lw : (int)
                line width in pixels
        """
        if isinstance(p1, vtki.vtkActor):
            p1 = p1.GetPosition()
            if isinstance(p0, vtki.vtkActor):
                p0 = p0.GetPosition()
        if isinstance(p0, Points):
            p0 = p0.coordinates

        # detect if user is passing a 2D list of points as p0=xlist, p1=ylist:
        if len(p0) > 3:
            if not utils.is_sequence(p0[0]) and not utils.is_sequence(p1[0]) and len(p0) == len(p1):
                # assume input is 2D xlist, ylist
                p0 = np.stack((p0, p1), axis=1)
                p1 = None
            p0 = utils.make3d(p0)
            if closed:
                p0 = np.append(p0, [p0[0]], axis=0)

        if p1 is not None:  # assume passing p0=[x,y]
            if len(p0) == 2 and not utils.is_sequence(p0[0]):
                p0 = (p0[0], p0[1], 0)
            if len(p1) == 2 and not utils.is_sequence(p1[0]):
                p1 = (p1[0], p1[1], 0)

        # detect if user is passing a list of points:
        if utils.is_sequence(p0[0]):
            listp = p0
        else:  # or just 2 points to link
            listp = [p0, p1]

        listp = np.array(listp)
        if listp.shape[1] == 2:
            listp = np.c_[listp, np.zeros(listp.shape[0])]

        xmn = np.min(listp, axis=0)
        xmx = np.max(listp, axis=0)
        dlen = np.linalg.norm(xmx - xmn) * np.clip(spacing, 0.01, 1.0) / 10
        if not dlen:
            super().__init__(vtki.vtkPolyData(), c, alpha)
            self.name = "DashedLine (void)"
            return

        qs = []
        for ipt in range(len(listp) - 1):
            p0 = listp[ipt]
            p1 = listp[ipt + 1]
            v = p1 - p0
            vdist = np.linalg.norm(v)
            n1 = int(vdist / dlen)
            if not n1:
                continue

            res = 0.0
            for i in range(n1 + 2):
                ist = (i - 0.5) / n1
                ist = max(ist, 0)
                qi = p0 + v * (ist - res / vdist)
                if ist > 1:
                    qi = p1
                    res = np.linalg.norm(qi - p1)
                    qs.append(qi)
                    break
                qs.append(qi)

        polylns = vtki.new("AppendPolyData")
        for i, q1 in enumerate(qs):
            if not i % 2:
                continue
            q0 = qs[i - 1]
            line_source = vtki.new("LineSource")
            line_source.SetPoint1(q0)
            line_source.SetPoint2(q1)
            line_source.Update()
            polylns.AddInputData(line_source.GetOutput())
        polylns.Update()

        super().__init__(polylns.GetOutput(), c, alpha)
        self.lw(lw).lighting("off")
        self.base = listp[0]
        if closed:
            self.top = listp[-2]
        else:
            self.top = listp[-1]
        self.name = "DashedLine"


class RoundedLine(Mesh):
    """
    Create a 2D line of specified thickness (in absolute units) passing through
    a list of input points. Borders of the line are rounded.
    """

    def __init__(self, pts, lw, res=10, c="gray4", alpha=1.0) -> None:
        """
        Arguments:
            pts : (list)
                a list of points in 2D or 3D (z will be ignored).
            lw : (float)
                thickness of the line.
            res : (int)
                resolution of the rounded regions

        Example:
            ```python
            from vedo import *
            pts = [(-4,-3),(1,1),(2,4),(4,1),(3,-1),(2,-5),(9,-3)]
            ln = Line(pts).z(0.01)
            ln.color("red5").linewidth(2)
            rl = RoundedLine(pts, 0.6)
            show(Points(pts), ln, rl, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/rounded_line.png)
        """
        pts = utils.make3d(pts)

        def _getpts(pts, revd=False):

            if revd:
                pts = list(reversed(pts))

            if len(pts) == 2:
                p0, p1 = pts
                v = p1 - p0
                dv = np.linalg.norm(v)
                nv = np.cross(v, (0, 0, -1))
                nv = nv / np.linalg.norm(nv) * lw
                return [p0 + nv, p1 + nv]

            ptsnew = []
            for k in range(len(pts) - 2):
                p0 = pts[k]
                p1 = pts[k + 1]
                p2 = pts[k + 2]
                v = p1 - p0
                u = p2 - p1
                du = np.linalg.norm(u)
                dv = np.linalg.norm(v)
                nv = np.cross(v, (0, 0, -1))
                nv = nv / np.linalg.norm(nv) * lw
                nu = np.cross(u, (0, 0, -1))
                nu = nu / np.linalg.norm(nu) * lw
                uv = np.cross(u, v)
                if k == 0:
                    ptsnew.append(p0 + nv)
                if uv[2] <= 0:
                    # the following computation can return a value
                    # ever so slightly > 1.0 causing arccos to fail.
                    uv_arg = np.dot(u, v) / du / dv
                    if uv_arg > 1.0:
                        # since the argument to arcos is 1, simply
                        # assign alpha to 0.0 without calculating the
                        # arccos
                        alpha = 0.0
                    else:
                        alpha = np.arccos(uv_arg)
                    db = lw * np.tan(alpha / 2)
                    p1new = p1 + nv - v / dv * db
                    ptsnew.append(p1new)
                else:
                    p1a = p1 + nv
                    p1b = p1 + nu
                    for i in range(0, res + 1):
                        pab = p1a * (res - i) / res + p1b * i / res
                        vpab = pab - p1
                        vpab = vpab / np.linalg.norm(vpab) * lw
                        ptsnew.append(p1 + vpab)
                if k == len(pts) - 3:
                    ptsnew.append(p2 + nu)
                    if revd:
                        ptsnew.append(p2 - nu)
            return ptsnew

        ptsnew = _getpts(pts) + _getpts(pts, revd=True)

        ppoints = vtki.vtkPoints()  # Generate the polyline
        ppoints.SetData(utils.numpy2vtk(np.asarray(ptsnew), dtype=np.float32))
        lines = vtki.vtkCellArray()
        npt = len(ptsnew)
        lines.InsertNextCell(npt)
        for i in range(npt):
            lines.InsertCellPoint(i)
        poly = vtki.vtkPolyData()
        poly.SetPoints(ppoints)
        poly.SetLines(lines)
        vct = vtki.new("ContourTriangulator")
        vct.SetInputData(poly)
        vct.Update()

        super().__init__(vct.GetOutput(), c, alpha)
        self.flat()
        self.properties.LightingOff()
        self.name = "RoundedLine"
        self.base = ptsnew[0]
        self.top = ptsnew[-1]


class Lines(Mesh):
    """
    Build the line segments between two lists of points `start_pts` and `end_pts`.
    `start_pts` can be also passed in the form `[[point1, point2], ...]`.
    """

    def __init__(
        self, start_pts, end_pts=None, dotted=False, res=1, scale=1.0, lw=1, c="k4", alpha=1.0
    ) -> None:
        """
        Arguments:
            scale : (float)
                apply a rescaling factor to the lengths.
            c : (color, int, str, list)
                color name, number, or list of [R,G,B] colors
            alpha : (float)
                opacity in range [0,1]
            lw : (int)
                line width in pixel units
            dotted : (bool)
                draw a dotted line
            res : (int)
                resolution, number of points along the line
                (only relevant if only 2 points are specified)

        Examples:
            - [fitspheres2.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/fitspheres2.py)

            ![](https://user-images.githubusercontent.com/32848391/52503049-ac9cb600-2be4-11e9-86af-72a538af14ef.png)
        """

        if isinstance(start_pts, vtki.vtkPolyData):########
            super().__init__(start_pts, c, alpha)
            self.lw(lw).lighting("off")
            self.name = "Lines"
            return ########################################

        if utils.is_sequence(start_pts) and len(start_pts)>1 and isinstance(start_pts[0], Line):
            # passing a list of Line, see tests/issues/issue_950.py
            polylns = vtki.new("AppendPolyData")
            for ln in start_pts:
                polylns.AddInputData(ln.dataset)
            polylns.Update()

            super().__init__(polylns.GetOutput(), c, alpha)
            self.lw(lw).lighting("off")
            if dotted:
                self.properties.SetLineStipplePattern(0xF0F0)
                self.properties.SetLineStippleRepeatFactor(1)
            self.name = "Lines"
            return ########################################

        if isinstance(start_pts, Points):
            start_pts = start_pts.coordinates
        if isinstance(end_pts, Points):
            end_pts = end_pts.coordinates

        if end_pts is not None:
            start_pts = np.stack((start_pts, end_pts), axis=1)

        polylns = vtki.new("AppendPolyData")

        if not utils.is_ragged(start_pts):

            for twopts in start_pts:
                line_source = vtki.new("LineSource")
                line_source.SetResolution(res)
                if len(twopts[0]) == 2:
                    line_source.SetPoint1(twopts[0][0], twopts[0][1], 0.0)
                else:
                    line_source.SetPoint1(twopts[0])

                if scale == 1:
                    pt2 = twopts[1]
                else:
                    vers = (np.array(twopts[1]) - twopts[0]) * scale
                    pt2 = np.array(twopts[0]) + vers

                if len(pt2) == 2:
                    line_source.SetPoint2(pt2[0], pt2[1], 0.0)
                else:
                    line_source.SetPoint2(pt2)
                polylns.AddInputConnection(line_source.GetOutputPort())

        else:

            polylns = vtki.new("AppendPolyData")
            for t in start_pts:
                t = utils.make3d(t)
                ppoints = vtki.vtkPoints()  # Generate the polyline
                ppoints.SetData(utils.numpy2vtk(t, dtype=np.float32))
                lines = vtki.vtkCellArray()
                npt = len(t)
                lines.InsertNextCell(npt)
                for i in range(npt):
                    lines.InsertCellPoint(i)
                poly = vtki.vtkPolyData()
                poly.SetPoints(ppoints)
                poly.SetLines(lines)
                polylns.AddInputData(poly)

        polylns.Update()

        super().__init__(polylns.GetOutput(), c, alpha)
        self.lw(lw).lighting("off")
        if dotted:
            self.properties.SetLineStipplePattern(0xF0F0)
            self.properties.SetLineStippleRepeatFactor(1)

        self.name = "Lines"


class Arc(Line):
    """
    Build a 2D circular arc between 2 points.
    """

    def __init__(
        self,
        center=None,
        point1=None,
        point2=None,
        normal=None,
        angle=None,
        invert=False,
        res=60,
        c="k3",
        alpha=1.0,
    ) -> None:
        """
        Build a 2D circular arc between 2 points.
        Two modes are available:
            1. [center, point1, point2] are specified

            2. [point1, normal, angle] are specified.

        In the first case it creates an arc defined by two endpoints and a center.
        In the second the arc spans the shortest angular sector defined by
        a starting point, a normal and a spanning angle.
        if `invert=True`, then the opposite happens.

        Example 1:
        ```python
        from vedo import *
        center = [0,1,0]
        p1 = [1,2,0.4]
        p2 = [0.5,3,-1]
        arc = Arc(center, p1, p2).lw(5).c("purple5")
        line2 = Line(center, p2)
        pts = Points([center, p1,p2], r=9, c='r')
        show(pts, line2, arc, f"length={arc.length()}", axes=1).close()
        ```

        Example 2:
        ```python
        from vedo import *
        arc = Arc(point1=[0,1,0], normal=[0,0,1], angle=270)
        arc.lw(5).c("purple5")
        origin = Point([0,0,0], r=9, c='r')
        show(origin, arc, arc.labels2d(), axes=1).close()
        ```
        """
        ar = vtki.new("ArcSource")
        if point2 is not None:
            center = utils.make3d(center)
            point1 = utils.make3d(point1)
            point2 = utils.make3d(point2)
            ar.UseNormalAndAngleOff()
            ar.SetPoint1(point1-center)
            ar.SetPoint2(point2-center)
        elif normal is not None and angle and point1 is not None:
            normal = utils.make3d(normal)
            point1 = utils.make3d(point1)
            ar.UseNormalAndAngleOn()
            ar.SetAngle(angle)
            ar.SetPolarVector(point1)
            ar.SetNormal(normal)
            self.top = normal
        else:
            vedo.logger.error("in Arc(), incorrect input combination.")
            raise TypeError
        ar.SetNegative(invert)
        ar.SetResolution(res)
        ar.Update()

        super().__init__(ar.GetOutput(), c, alpha)
        self.lw(2).lighting("off")
        if point2 is not None: # nb: not center
            self.pos(center)
        self.name = "Arc"


class Spline(Line):
    """
    Find the B-Spline curve through a set of points. This curve does not necessarily
    pass exactly through all the input points. Needs to import `scipy`.
    """

    def __init__(self, points, smooth=0.0, degree=2, closed=False, res=None, easing="") -> None:
        """
        Arguments:
            smooth : (float)
                smoothing factor.
                - 0 = interpolate points exactly [default].
                - 1 = average point positions.
            degree : (int)
                degree of the spline (between 1 and 5).
            easing : (str)
                control sensity of points along the spline.
                Available options are
                `[InSine, OutSine, Sine, InQuad, OutQuad, InCubic, OutCubic, InQuart, OutQuart, InCirc, OutCirc].`
                Can be used to create animations (move objects at varying speed).
                See e.g.: https://easings.net
            res : (int)
                number of points on the spline

        See also: `CSpline` and `KSpline`.

        Examples:
            - [spline_ease.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/spline_ease.py)

                ![](https://vedo.embl.es/images/simulations/spline_ease.gif)
        """
        from scipy.interpolate import splprep, splev

        if isinstance(points, Points):
            points = points.coordinates

        points = utils.make3d(points)

        per = 0
        if closed:
            points = np.append(points, [points[0]], axis=0)
            per = 1

        if res is None:
            res = len(points) * 10

        points = np.array(points, dtype=float)

        minx, miny, minz = np.min(points, axis=0)
        maxx, maxy, maxz = np.max(points, axis=0)
        maxb = max(maxx - minx, maxy - miny, maxz - minz)
        smooth *= maxb / 2  # must be in absolute units

        x = np.linspace(0.0, 1.0, res)
        if easing:
            if easing == "InSine":
                x = 1.0 - np.cos((x * np.pi) / 2)
            elif easing == "OutSine":
                x = np.sin((x * np.pi) / 2)
            elif easing == "Sine":
                x = -(np.cos(np.pi * x) - 1) / 2
            elif easing == "InQuad":
                x = x * x
            elif easing == "OutQuad":
                x = 1.0 - (1 - x) * (1 - x)
            elif easing == "InCubic":
                x = x * x
            elif easing == "OutCubic":
                x = 1.0 - np.power(1 - x, 3)
            elif easing == "InQuart":
                x = x * x * x * x
            elif easing == "OutQuart":
                x = 1.0 - np.power(1 - x, 4)
            elif easing == "InCirc":
                x = 1.0 - np.sqrt(1 - np.power(x, 2))
            elif easing == "OutCirc":
                x = np.sqrt(1.0 - np.power(x - 1, 2))
            else:
                vedo.logger.error(f"unknown ease mode {easing}")

        # find the knots
        tckp, _ = splprep(points.T, task=0, s=smooth, k=degree, per=per)
        # evaluate spLine, including interpolated points:
        xnew, ynew, znew = splev(x, tckp)

        super().__init__(np.c_[xnew, ynew, znew], lw=2)
        self.name = "Spline"


class KSpline(Line):
    """
    Return a [Kochanek spline](https://en.wikipedia.org/wiki/Kochanek%E2%80%93Bartels_spline)
    which runs exactly through all the input points.
    """

    def __init__(self, points,
                 continuity=0.0, tension=0.0, bias=0.0, closed=False, res=None) -> None:
        """
        Arguments:
            continuity : (float)
                changes the sharpness in change between tangents
            tension : (float)
                changes the length of the tangent vector
            bias : (float)
                changes the direction of the tangent vector
            closed : (bool)
                join last to first point to produce a closed curve
            res : (int)
                approximate resolution of the output line.
                Default is 20 times the number of input points.

        ![](https://user-images.githubusercontent.com/32848391/65975805-73fd6580-e46f-11e9-8957-75eddb28fa72.png)

        Warning:
            This class is not necessarily generating the exact number of points
            as requested by `res`. Some points may be concident and removed.

        See also: `Spline` and `CSpline`.
        """
        if isinstance(points, Points):
            points = points.coordinates

        if not res:
            res = len(points) * 20

        points = utils.make3d(points).astype(float)

        vtkKochanekSpline = vtki.get_class("KochanekSpline")
        xspline = vtkKochanekSpline()
        yspline = vtkKochanekSpline()
        zspline = vtkKochanekSpline()
        for s in [xspline, yspline, zspline]:
            if bias:
                s.SetDefaultBias(bias)
            if tension:
                s.SetDefaultTension(tension)
            if continuity:
                s.SetDefaultContinuity(continuity)
            s.SetClosed(closed)

        lenp = len(points[0]) > 2

        for i, p in enumerate(points):
            xspline.AddPoint(i, p[0])
            yspline.AddPoint(i, p[1])
            if lenp:
                zspline.AddPoint(i, p[2])

        ln = []
        for pos in np.linspace(0, len(points), res):
            x = xspline.Evaluate(pos)
            y = yspline.Evaluate(pos)
            z = 0
            if lenp:
                z = zspline.Evaluate(pos)
            ln.append((x, y, z))

        super().__init__(ln, lw=2)
        self.clean()
        self.lighting("off")
        self.name = "KSpline"
        self.base = np.array(points[0], dtype=float)
        self.top = np.array(points[-1], dtype=float)


class CSpline(Line):
    """
    Return a Cardinal spline which runs exactly through all the input points.
    """

    def __init__(self, points, closed=False, res=None) -> None:
        """
        Arguments:
            closed : (bool)
                join last to first point to produce a closed curve
            res : (int)
                approximate resolution of the output line.
                Default is 20 times the number of input points.

        Warning:
            This class is not necessarily generating the exact number of points
            as requested by `res`. Some points may be concident and removed.

        See also: `Spline` and `KSpline`.
        """

        if isinstance(points, Points):
            points = points.coordinates

        if not res:
            res = len(points) * 20

        points = utils.make3d(points).astype(float)

        vtkCardinalSpline = vtki.get_class("CardinalSpline")
        xspline = vtkCardinalSpline()
        yspline = vtkCardinalSpline()
        zspline = vtkCardinalSpline()
        for s in [xspline, yspline, zspline]:
            s.SetClosed(closed)

        lenp = len(points[0]) > 2

        for i, p in enumerate(points):
            xspline.AddPoint(i, p[0])
            yspline.AddPoint(i, p[1])
            if lenp:
                zspline.AddPoint(i, p[2])

        ln = []
        for pos in np.linspace(0, len(points), res):
            x = xspline.Evaluate(pos)
            y = yspline.Evaluate(pos)
            z = 0
            if lenp:
                z = zspline.Evaluate(pos)
            ln.append((x, y, z))

        super().__init__(ln, lw=2)
        self.clean()
        self.lighting("off")
        self.name = "CSpline"
        self.base = points[0]
        self.top = points[-1]


class Bezier(Line):
    """
    Generate the Bezier line that links the first to the last point.
    """

    def __init__(self, points, res=None) -> None:
        """
        Example:
            ```python
            from vedo import *
            import numpy as np
            pts = np.random.randn(25,3)
            for i,p in enumerate(pts):
                p += [5*i, 15*sin(i/2), i*i*i/200]
            show(Points(pts), Bezier(pts), axes=1).close()
            ```
            ![](https://user-images.githubusercontent.com/32848391/90437534-dafd2a80-e0d2-11ea-9b93-9ecb3f48a3ff.png)
        """
        N = len(points)
        if res is None:
            res = 10 * N
        t = np.linspace(0, 1, num=res)
        bcurve = np.zeros((res, len(points[0])))

        def binom(n, k):
            b = 1
            for t in range(1, min(k, n - k) + 1):
                b *= n / t
                n -= 1
            return b

        def bernstein(n, k):
            coeff = binom(n, k)

            def _bpoly(x):
                return coeff * x ** k * (1 - x) ** (n - k)

            return _bpoly

        for ii in range(N):
            b = bernstein(N - 1, ii)(t)
            bcurve += np.outer(b, points[ii])
        super().__init__(bcurve, lw=2)
        self.name = "BezierLine"


class NormalLines(Mesh):
    """
    Build an `Glyph` to show the normals at cell centers or at mesh vertices.

    Arguments:
        ratio : (int)
            show 1 normal every `ratio` cells.
        on : (str)
            either "cells" or "points".
        scale : (float)
            scale factor to control size.
    """

    def __init__(self, msh, ratio=1, on="cells", scale=1.0) -> None:

        poly = msh.clone().dataset

        if "cell" in on:
            centers = vtki.new("CellCenters")
            centers.SetInputData(poly)
            centers.Update()
            poly = centers.GetOutput()

        mask_pts = vtki.new("MaskPoints")
        mask_pts.SetInputData(poly)
        mask_pts.SetOnRatio(ratio)
        mask_pts.RandomModeOff()
        mask_pts.Update()

        ln = vtki.new("LineSource")
        ln.SetPoint1(0, 0, 0)
        ln.SetPoint2(1, 0, 0)
        ln.Update()
        glyph = vtki.vtkGlyph3D()
        glyph.SetSourceData(ln.GetOutput())
        glyph.SetInputData(mask_pts.GetOutput())
        glyph.SetVectorModeToUseNormal()

        b = poly.GetBounds()
        f = max([b[1] - b[0], b[3] - b[2], b[5] - b[4]]) / 50 * scale
        glyph.SetScaleFactor(f)
        glyph.OrientOn()
        glyph.Update()

        super().__init__(glyph.GetOutput())

        self.actor.PickableOff()
        prop = vtki.vtkProperty()
        prop.DeepCopy(msh.properties)
        self.actor.SetProperty(prop)
        self.properties = prop
        self.properties.LightingOff()
        self.mapper.ScalarVisibilityOff()
        self.name = "NormalLines"


class Tube(Mesh):
    """
    Build a tube along the line defined by a set of points.
    """

    def __init__(self, points, r=1.0, cap=True, res=12, c=None, alpha=1.0) -> None:
        """
        Arguments:
            r :  (float, list)
                constant radius or list of radii.
            res : (int)
                resolution, number of the sides of the tube
            c : (color)
                constant color or list of colors for each point.

        Example:
            Create a tube along a line, with data associated to each point:

            ```python
            from vedo import *
            line = Line([(0,0,0), (1,1,1), (2,0,1), (3,1,0)]).lw(5)
            scalars = np.array([0, 1, 2, 3])
            line.pointdata["myscalars"] = scalars
            tube = Tube(line, r=0.1).lw(1)
            tube.cmap('viridis', "myscalars").add_scalarbar3d()
            show(line, tube, axes=1).close()
            ```

        Examples:
            - [ribbon.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/ribbon.py)
            - [tube_radii.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/tube_radii.py)

                ![](https://vedo.embl.es/images/basic/tube.png)
        """
        if utils.is_sequence(points):
            vpoints = vtki.vtkPoints()
            idx = len(points)
            for p in points:
                vpoints.InsertNextPoint(p)
            line = vtki.new("PolyLine")
            line.GetPointIds().SetNumberOfIds(idx)
            for i in range(idx):
                line.GetPointIds().SetId(i, i)
            lines = vtki.vtkCellArray()
            lines.InsertNextCell(line)
            polyln = vtki.vtkPolyData()
            polyln.SetPoints(vpoints)
            polyln.SetLines(lines)
            self.base = np.asarray(points[0], dtype=float)
            self.top = np.asarray(points[-1], dtype=float)

        elif isinstance(points, Mesh):
            polyln = points.dataset
            n = polyln.GetNumberOfPoints()
            self.base = np.array(polyln.GetPoint(0))
            self.top = np.array(polyln.GetPoint(n - 1))

        # from vtkmodules.vtkFiltersCore import vtkTubeBender
        # bender = vtkTubeBender()
        # bender.SetInputData(polyln)
        # bender.SetRadius(r)
        # bender.Update()
        # polyln = bender.GetOutput()

        tuf = vtki.new("TubeFilter")
        tuf.SetCapping(cap)
        tuf.SetNumberOfSides(res)
        tuf.SetInputData(polyln)
        if utils.is_sequence(r):
            arr = utils.numpy2vtk(r, dtype=float)
            arr.SetName("TubeRadius")
            polyln.GetPointData().AddArray(arr)
            polyln.GetPointData().SetActiveScalars("TubeRadius")
            tuf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        else:
            tuf.SetRadius(r)

        usingColScals = False
        if utils.is_sequence(c):
            usingColScals = True
            cc = vtki.vtkUnsignedCharArray()
            cc.SetName("TubeColors")
            cc.SetNumberOfComponents(3)
            cc.SetNumberOfTuples(len(c))
            for i, ic in enumerate(c):
                r, g, b = get_color(ic)
                cc.InsertTuple3(i, int(255 * r), int(255 * g), int(255 * b))
            polyln.GetPointData().AddArray(cc)
            c = None
        tuf.Update()

        super().__init__(tuf.GetOutput(), c, alpha)
        self.phong()
        if usingColScals:
            self.mapper.SetScalarModeToUsePointFieldData()
            self.mapper.ScalarVisibilityOn()
            self.mapper.SelectColorArray("TubeColors")
            self.mapper.Modified()
        self.name = "Tube"


def ThickTube(pts, r1, r2, res=12, c=None, alpha=1.0) -> Mesh | None:
    """
    Create a tube with a thickness along a line of points.

    Example:
    ```python
    from vedo import *
    pts = [[sin(x), cos(x), x/3] for x in np.arange(0.1, 3, 0.3)]
    vline = Line(pts, lw=5, c='red5')
    thick_tube = ThickTube(vline, r1=0.2, r2=0.3).lw(1)
    show(vline, thick_tube, axes=1).close()
    ```
    ![](https://vedo.embl.es/images/feats/thick_tube.png)
    """

    def make_cap(t1, t2):
        newpoints = t1.coordinates.tolist() + t2.coordinates.tolist()
        newfaces = []
        for i in range(n - 1):
            newfaces.append([i, i + 1, i + n])
            newfaces.append([i + n, i + 1, i + n + 1])
        newfaces.append([2 * n - 1, 0, n])
        newfaces.append([2 * n - 1, n - 1, 0])
        capm = utils.buildPolyData(newpoints, newfaces)
        return capm

    assert r1 < r2

    t1 = Tube(pts, r=r1, cap=False, res=res)
    t2 = Tube(pts, r=r2, cap=False, res=res)

    tc1a, tc1b = t1.boundaries().split()
    tc2a, tc2b = t2.boundaries().split()
    n = tc1b.npoints

    tc1b.join(reset=True).clean()  # needed because indices are flipped
    tc2b.join(reset=True).clean()

    capa = make_cap(tc1a, tc2a)
    capb = make_cap(tc1b, tc2b)

    thick_tube = merge(t1, t2, capa, capb)
    if thick_tube:
        thick_tube.c(c).alpha(alpha)
        thick_tube.base = t1.base
        thick_tube.top  = t1.top
        thick_tube.name = "ThickTube"
        return thick_tube
    return None


class Tubes(Mesh):
    """
    Build tubes around a `Lines` object.
    """
    def __init__(
            self,
            lines,
            r=1,
            vary_radius_by_scalar=False,
            vary_radius_by_vector=False,
            vary_radius_by_vector_norm=False,
            vary_radius_by_absolute_scalar=False,
            max_radius_factor=100,
            cap=True,
            res=12
        ) -> None:
        """
        Wrap tubes around the input `Lines` object.

        Arguments:
            lines : (Lines)
                input Lines object.
            r : (float)
                constant radius
            vary_radius_by_scalar : (bool)
                use scalar array to control radius
            vary_radius_by_vector : (bool)
                use vector array to control radius
            vary_radius_by_vector_norm : (bool)
                use vector norm to control radius
            vary_radius_by_absolute_scalar : (bool)
                use absolute scalar value to control radius
            max_radius_factor : (float)
                max tube radius as a multiple of the min radius
            cap : (bool)
                capping of the tube
            res : (int)
                resolution, number of the sides of the tube
            c : (color)
                constant color or list of colors for each point.

        Examples:
            - [streamlines1.py](https://github.com/marcomusy/vedo/blob/master/examples/volumetric/streamlines1.py)
        """
        plines = lines.dataset
        if plines.GetNumberOfLines() == 0:
            vedo.logger.warning("Tubes(): input Lines is empty.")

        tuf = vtki.new("TubeFilter")
        if vary_radius_by_scalar:
            tuf.SetVaryRadiusToVaryRadiusByScalar()
        elif vary_radius_by_vector:
            tuf.SetVaryRadiusToVaryRadiusByVector()
        elif vary_radius_by_vector_norm:
            tuf.SetVaryRadiusToVaryRadiusByVectorNorm()
        elif vary_radius_by_absolute_scalar:
            tuf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tuf.SetRadius(r)
        tuf.SetCapping(cap)
        tuf.SetGenerateTCoords(0)
        tuf.SetSidesShareVertices(1)
        tuf.SetRadiusFactor(max_radius_factor)
        tuf.SetNumberOfSides(res)
        tuf.SetInputData(plines)
        tuf.Update()

        super().__init__(tuf.GetOutput())
        self.name = "Tubes"

