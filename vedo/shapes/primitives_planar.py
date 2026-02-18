#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Planar and near-planar primitive shapes."""

from typing import Any
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import settings, utils
from vedo.core.transformations import LinearTransform, pol2cart, cart2spher, spher2cart
from vedo.colors import get_color, printc
from vedo.mesh import Mesh
from vedo.pointcloud import Points, merge
from vedo.grids.image import Image

class Triangle(Mesh):
    """Create a triangle from 3 points in space."""

    def __init__(self, p1, p2, p3, c="green7", alpha=1.0) -> None:
        """Create a triangle from 3 points in space."""
        super().__init__([[p1, p2, p3], [[0, 1, 2]]], c, alpha)
        self.properties.LightingOff()
        self.name = "Triangle"


class Polygon(Mesh):
    """
    Build a polygon in the `xy` plane.
    """

    def __init__(self, pos=(0, 0, 0), nsides=6, r=1.0, c="coral", alpha=1.0) -> None:
        """
        Build a polygon in the `xy` plane of `nsides` of radius `r`.

        ![](https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestRegularPolygonSource.png)
        """
        t = np.linspace(np.pi / 2, 5 / 2 * np.pi, num=nsides, endpoint=False)
        pts = pol2cart(np.ones_like(t) * r, t).T
        faces = [list(range(nsides))]
        # do not use: vtkRegularPolygonSource
        super().__init__([pts, faces], c, alpha)
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.pos(pos)
        self.properties.LightingOff()
        self.name = "Polygon " + str(nsides)


class Circle(Polygon):
    """
    Build a Circle of radius `r`.
    """

    def __init__(self, pos=(0, 0, 0), r=1.0, res=120, c="gray5", alpha=1.0) -> None:
        """
        Build a Circle of radius `r`.
        """
        super().__init__(pos, nsides=res, r=r)

        self.nr_of_points = 0
        self.va = 0
        self.vb = 0
        self.axis1: list[float] = []
        self.axis2: list[float] = []
        self.center: list[float] = []  # filled by pointcloud.pca_ellipse()
        self.pvalue = 0.0              # filled by pointcloud.pca_ellipse()
        self.alpha(alpha).c(c)
        self.name = "Circle"

    def acircularity(self) -> float:
        """
        Return a measure of how different an ellipse is from a circle.
        Values close to zero correspond to a circular object.
        """
        a, b = self.va, self.vb
        value = 0.0
        if a+b:
            value = ((a-b)/(a+b))**2
        return value

class GeoCircle(Polygon):
    """
    Build a Circle of radius `r`.
    """

    def __init__(self, lat, lon, r=1.0, res=60, c="red4", alpha=1.0) -> None:
        """
        Build a Circle of radius `r` as projected on a geographic map.
        Circles near the poles will look very squashed.

        See example:
            ```bash
            vedo -r earthquake
            ```
        """
        coords = []
        sinr, cosr = np.sin(r), np.cos(r)
        sinlat, coslat = np.sin(lat), np.cos(lat)
        for phi in np.linspace(0, 2 * np.pi, num=res, endpoint=False):
            clat = np.arcsin(sinlat * cosr + coslat * sinr * np.cos(phi))
            clng = lon + np.arctan2(np.sin(phi) * sinr * coslat, cosr - sinlat * np.sin(clat))
            coords.append([clng / np.pi + 1, clat * 2 / np.pi + 1, 0])

        super().__init__(nsides=res, c=c, alpha=alpha)
        self.coordinates = coords # warp polygon points to match geo projection
        self.name = "Circle"


class Star(Mesh):
    """
    Build a 2D star shape.
    """

    def __init__(self, pos=(0, 0, 0), n=5, r1=0.7, r2=1.0, line=False, c="blue6", alpha=1.0) -> None:
        """
        Build a 2D star shape of `n` cusps of inner radius `r1` and outer radius `r2`.

        If line is True then only build the outer line (no internal surface meshing).

        Example:
            - [extrude1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/extrude1.py)

                ![](https://vedo.embl.es/images/basic/extrude.png)
        """
        t = np.linspace(np.pi / 2, 5 / 2 * np.pi, num=n, endpoint=False)
        x, y = pol2cart(np.ones_like(t) * r2, t)
        pts = np.c_[x, y, np.zeros_like(x)]

        apts = []
        for i, p in enumerate(pts):
            apts.append(p)
            if i + 1 < n:
                apts.append((p + pts[i + 1]) / 2 * r1 / r2)
        apts.append((pts[-1] + pts[0]) / 2 * r1 / r2)

        if line:
            apts.append(pts[0])
            poly = utils.buildPolyData(apts, lines=[list(range(len(apts)))])
            super().__init__(poly, c, alpha)
            self.lw(2)
        else:
            apts.append((0, 0, 0))
            cells = []
            for i in range(2 * n - 1):
                cell = [2 * n, i, i + 1]
                cells.append(cell)
            cells.append([2 * n, i + 1, 0])
            super().__init__([apts, cells], c, alpha)

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        self.properties.LightingOff()
        self.name = "Star"


class Disc(Mesh):
    """
    Build a 2D disc.
    """

    def __init__(
        self, pos=(0, 0, 0), r1=0.5, r2=1.0, res=(1, 120), angle_range=(), c="gray4", alpha=1.0
    ) -> None:
        """
        Build a 2D disc of inner radius `r1` and outer radius `r2`.

        Set `res` as the resolution in R and Phi (can be a list).

        Use `angle_range` to create a disc sector between the 2 specified angles.

        ![](https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestDisk.png)
        """
        if utils.is_sequence(res):
            res_r, res_phi = res
        else:
            res_r, res_phi = res, 12 * res

        if len(angle_range) == 0:
            ps = vtki.new("DiskSource")
        else:
            ps = vtki.new("SectorSource")
            ps.SetStartAngle(angle_range[0])
            ps.SetEndAngle(angle_range[1])

        ps.SetInnerRadius(r1)
        ps.SetOuterRadius(r2)
        ps.SetRadialResolution(res_r)
        ps.SetCircumferentialResolution(res_phi)
        ps.Update()
        super().__init__(ps.GetOutput(), c, alpha)
        self.flat()
        self.pos(utils.make3d(pos))
        self.name = "Disc"

class IcoSphere(Mesh):
    """
    Create a sphere made of a uniform triangle mesh.
    """

    def __init__(self, pos=(0, 0, 0), r=1.0, subdivisions=4, c="r5", alpha=1.0) -> None:
        """
        Create a sphere made of a uniform triangle mesh
        (from recursive subdivision of an icosahedron).

        Example:
        ```python
        from vedo import *
        icos = IcoSphere(subdivisions=3)
        icos.compute_quality().cmap('coolwarm')
        icos.show(axes=1).close()
        ```
        ![](https://vedo.embl.es/images/basic/icosphere.jpg)
        """
        subdivisions = int(min(subdivisions, 9))  # to avoid disasters

        t = (1.0 + np.sqrt(5.0)) / 2.0
        points = np.array(
            [
                [-1, t, 0],
                [1, t, 0],
                [-1, -t, 0],
                [1, -t, 0],
                [0, -1, t],
                [0, 1, t],
                [0, -1, -t],
                [0, 1, -t],
                [t, 0, -1],
                [t, 0, 1],
                [-t, 0, -1],
                [-t, 0, 1],
            ]
        )
        faces = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
        super().__init__([points * r, faces], c=c, alpha=alpha)

        for _ in range(subdivisions):
            self.subdivide(method=1)
            pts = utils.versor(self.coordinates) * r
            self.coordinates = pts

        self.pos(pos)
        self.name = "IcoSphere"


class Sphere(Mesh):
    """
    Build a sphere.
    """

    def __init__(self, pos=(0, 0, 0), r=1.0, res=24, quads=False, c="r5", alpha=1.0) -> None:
        """
        Build a sphere at position `pos` of radius `r`.

        Arguments:
            r : (float)
                sphere radius
            res : (int, list)
                resolution in phi, resolution in theta is by default `2*res`
            quads : (bool)
                sphere mesh will be made of quads instead of triangles

        [](https://user-images.githubusercontent.com/32848391/72433092-f0a31e00-3798-11ea-85f7-b2f5fcc31568.png)
        """
        if len(pos) == 2:
            pos = np.asarray([pos[0], pos[1], 0])

        self.radius = r  # used by fitSphere
        self.center = pos
        self.residue = 0

        if quads:
            res = max(res, 4)
            img = vtki.vtkImageData()
            img.SetDimensions(res - 1, res - 1, res - 1)
            rs = 1.0 / (res - 2)
            img.SetSpacing(rs, rs, rs)
            gf = vtki.new("GeometryFilter")
            gf.SetInputData(img)
            gf.Update()
            super().__init__(gf.GetOutput(), c, alpha)
            self.lw(0.1)

            cgpts = self.coordinates - (0.5, 0.5, 0.5)

            x, y, z = cgpts.T
            x = x * (1 + x * x) / 2
            y = y * (1 + y * y) / 2
            z = z * (1 + z * z) / 2
            _, theta, phi = cart2spher(x, y, z)

            pts = spher2cart(np.ones_like(phi) * r, theta, phi).T
            self.coordinates = pts

        else:
            if utils.is_sequence(res):
                res_t, res_phi = res
            else:
                res_t, res_phi = 2 * res, res

            ss = vtki.new("SphereSource")
            ss.SetRadius(r)
            ss.SetThetaResolution(res_t)
            ss.SetPhiResolution(res_phi)
            ss.Update()

            super().__init__(ss.GetOutput(), c, alpha)

        self.phong()
        self.pos(pos)
        self.name = "Sphere"


class Spheres(Mesh):
    """
    Build a large set of spheres.
    """

    def __init__(self, centers, r=1.0, res=8, c="red5", alpha=1) -> None:
        """
        Build a (possibly large) set of spheres at `centers` of radius `r`.

        Either `c` or `r` can be a list of RGB colors or radii.

        Examples:
            - [manyspheres.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/manyspheres.py)

            ![](https://vedo.embl.es/images/basic/manyspheres.jpg)
        """

        if isinstance(centers, Points):
            centers = centers.coordinates
        centers = np.asarray(centers, dtype=float)
        base = centers[0]

        cisseq = False
        if utils.is_sequence(c):
            cisseq = True

        if cisseq:
            if len(centers) != len(c):
                vedo.logger.error(f"mismatch #centers {len(centers)} != {len(c)} #colors")
                raise RuntimeError()

        risseq = False
        if utils.is_sequence(r):
            risseq = True

        if risseq:
            if len(centers) != len(r):
                vedo.logger.error(f"mismatch #centers {len(centers)} != {len(r)} #radii")
                raise RuntimeError()
        if cisseq and risseq:
            vedo.logger.error("Limitation: c and r cannot be both sequences.")
            raise RuntimeError()

        src = vtki.new("SphereSource")
        if not risseq:
            src.SetRadius(r)
        if utils.is_sequence(res):
            res_t, res_phi = res
        else:
            res_t, res_phi = 2 * res, res

        src.SetThetaResolution(res_t)
        src.SetPhiResolution(res_phi)
        src.Update()

        psrc = vtki.new("PointSource")
        psrc.SetNumberOfPoints(len(centers))
        psrc.Update()
        pd = psrc.GetOutput()
        vpts = pd.GetPoints()

        glyph = vtki.vtkGlyph3D()
        glyph.SetSourceConnection(src.GetOutputPort())

        if cisseq:
            glyph.SetColorModeToColorByScalar()
            ucols = vtki.vtkUnsignedCharArray()
            ucols.SetNumberOfComponents(3)
            ucols.SetName("Colors")
            for acol in c:
                cx, cy, cz = get_color(acol)
                ucols.InsertNextTuple3(cx * 255, cy * 255, cz * 255)
            pd.GetPointData().AddArray(ucols)
            pd.GetPointData().SetActiveScalars("Colors")
            glyph.ScalingOff()
        elif risseq:
            glyph.SetScaleModeToScaleByScalar()
            urads = utils.numpy2vtk(2 * np.ascontiguousarray(r), dtype=np.float32)
            urads.SetName("Radii")
            pd.GetPointData().AddArray(urads)
            pd.GetPointData().SetActiveScalars("Radii")

        vpts.SetData(utils.numpy2vtk(centers - base, dtype=np.float32))

        glyph.SetInputData(pd)
        glyph.Update()

        super().__init__(glyph.GetOutput(), alpha=alpha)
        self.pos(base)
        self.phong()
        if cisseq:
            self.mapper.ScalarVisibilityOn()
        else:
            self.mapper.ScalarVisibilityOff()
            self.c(c)
        self.name = "Spheres"


class Earth(Mesh):
    """
    Build a textured mesh representing the Earth.
    """

    def __init__(self, style=1, r=1.0) -> None:
        """
        Build a textured mesh representing the Earth.

        Example:
            - [geodesic_curve.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/geodesic_curve.py)

                ![](https://vedo.embl.es/images/advanced/geodesic.png)
        """
        tss = vtki.new("TexturedSphereSource")
        tss.SetRadius(r)
        tss.SetThetaResolution(72)
        tss.SetPhiResolution(36)
        tss.Update()
        super().__init__(tss.GetOutput(), c="w")
        atext = vtki.vtkTexture()
        pnm_reader = vtki.new("JPEGReader")
        fn = vedo.file_io.download(vedo.dataurl + f"textures/earth{style}.jpg", verbose=False)
        pnm_reader.SetFileName(fn)
        atext.SetInputConnection(pnm_reader.GetOutputPort())
        atext.InterpolateOn()
        self.texture(atext)
        self.name = "Earth"


class Ellipsoid(Mesh):
    """Build a 3D ellipsoid."""
    def __init__(
        self,
        pos=(0, 0, 0),
        axis1=(0.5, 0, 0),
        axis2=(0, 1, 0),
        axis3=(0, 0, 1.5),
        res=24,
        c="cyan4",
        alpha=1.0,
    ) -> None:
        """
        Build a 3D ellipsoid centered at position `pos`.

        Arguments:
            axis1 : (list)
                First axis. Length corresponds to semi-axis.
            axis2 : (list)
                Second axis. Length corresponds to semi-axis.
            axis3 : (list)
                Third axis. Length corresponds to semi-axis.
        """
        self.center = utils.make3d(pos)

        self.axis1 = utils.make3d(axis1)
        self.axis2 = utils.make3d(axis2)
        self.axis3 = utils.make3d(axis3)

        self.va = np.linalg.norm(self.axis1)
        self.vb = np.linalg.norm(self.axis2)
        self.vc = np.linalg.norm(self.axis3)

        self.va_error = 0
        self.vb_error = 0
        self.vc_error = 0

        self.nr_of_points = 1  # used by pointcloud.pca_ellipsoid()
        self.pvalue = 0        # used by pointcloud.pca_ellipsoid()

        if utils.is_sequence(res):
            res_t, res_phi = res
        else:
            res_t, res_phi = 2 * res, res

        elli_source = vtki.new("SphereSource")
        elli_source.SetRadius(1)
        elli_source.SetThetaResolution(res_t)
        elli_source.SetPhiResolution(res_phi)
        elli_source.Update()

        super().__init__(elli_source.GetOutput(), c, alpha)

        matrix = np.c_[self.axis1, self.axis2, self.axis3]
        lt = LinearTransform(matrix).translate(pos)
        self.apply_transform(lt)
        self.name = "Ellipsoid"

    def asphericity(self) -> float:
        """
        Return a measure of how different an ellipsoid is from a sphere.
        Values close to zero correspond to a spheric object.
        """
        a, b, c = self.va, self.vb, self.vc
        asp = ( ((a-b)/(a+b))**2
              + ((a-c)/(a+c))**2
              + ((b-c)/(b+c))**2 ) / 3. * 4.
        return float(asp)

    def asphericity_error(self) -> float:
        """
        Calculate statistical error on the asphericity value.

        Errors on the main axes are stored in
        `Ellipsoid.va_error`, Ellipsoid.vb_error` and `Ellipsoid.vc_error`.
        """
        a, b, c = self.va, self.vb, self.vc
        sqrtn = np.sqrt(self.nr_of_points)
        ea, eb, ec = a / 2 / sqrtn, b / 2 / sqrtn, b / 2 / sqrtn

        # from sympy import *
        # init_printing(use_unicode=True)
        # a, b, c, ea, eb, ec = symbols("a b c, ea, eb,ec")
        # L = (
        #    (((a - b) / (a + b)) ** 2 + ((c - b) / (c + b)) ** 2 + ((a - c) / (a + c)) ** 2)
        #    / 3 * 4)
        # dl2 = (diff(L, a) * ea) ** 2 + (diff(L, b) * eb) ** 2 + (diff(L, c) * ec) ** 2
        # print(dl2)
        # exit()

        dL2 = (
            ea ** 2
            * (
                -8 * (a - b) ** 2 / (3 * (a + b) ** 3)
                - 8 * (a - c) ** 2 / (3 * (a + c) ** 3)
                + 4 * (2 * a - 2 * c) / (3 * (a + c) ** 2)
                + 4 * (2 * a - 2 * b) / (3 * (a + b) ** 2)
            ) ** 2
            + eb ** 2
            * (
                4 * (-2 * a + 2 * b) / (3 * (a + b) ** 2)
                - 8 * (a - b) ** 2 / (3 * (a + b) ** 3)
                - 8 * (-b + c) ** 2 / (3 * (b + c) ** 3)
                + 4 * (2 * b - 2 * c) / (3 * (b + c) ** 2)
            ) ** 2
            + ec ** 2
            * (
                4 * (-2 * a + 2 * c) / (3 * (a + c) ** 2)
                - 8 * (a - c) ** 2 / (3 * (a + c) ** 3)
                + 4 * (-2 * b + 2 * c) / (3 * (b + c) ** 2)
                - 8 * (-b + c) ** 2 / (3 * (b + c) ** 3)
            ) ** 2
        )
        err = np.sqrt(dL2)
        self.va_error = ea
        self.vb_error = eb
        self.vc_error = ec
        return err


class Grid(Mesh):
    """
    An even or uneven 2D grid.
    """

    def __init__(self, pos=(0, 0, 0), s=(1, 1), res=(10, 10), lw=1, c="k3", alpha=1.0) -> None:
        """
        Create an even or uneven 2D grid.
        Can also be created from a `np.mgrid` object (see example).

        Arguments:
            pos : (list, Points, Mesh)
                position in space, can also be passed as a bounding box [xmin,xmax, ymin,ymax].
            s : (float, list)
                if a float is provided it is interpreted as the total size along x and y,
                if a list of coords is provided they are interpreted as the vertices of the grid along x and y.
                In this case keyword `res` is ignored (see example below).
            res : (list)
                resolutions along x and y, e.i. the number of subdivisions
            lw : (int)
                line width

        Example:
            ```python
            from vedo import *
            xcoords = np.arange(0, 2, 0.2)
            ycoords = np.arange(0, 1, 0.2)
            sqrtx = sqrt(xcoords)
            grid = Grid(s=(sqrtx, ycoords)).lw(2)
            grid.show(axes=8).close()

            # Can also create a grid from a np.mgrid:
            X, Y = np.mgrid[-12:12:10*1j, 200:215:10*1j]
            vgrid = Grid(s=(X[:,0], Y[0]))
            vgrid.show(axes=8).close()
            ```
            ![](https://vedo.embl.es/images/feats/uneven_grid.png)
        """
        resx, resy = res
        sx, sy = s

        try:
            bb = pos.bounds()
            pos = [(bb[0] + bb[1])/2, (bb[2] + bb[3])/2, (bb[4] + bb[5])/2]
            sx = bb[1] - bb[0]
            sy = bb[3] - bb[2]
        except AttributeError:
            pass

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        elif len(pos) in [4,6]: # passing a bounding box
            bb = pos
            pos = [(bb[0] + bb[1])/2, (bb[2] + bb[3])/2, 0]
            sx = bb[1] - bb[0]
            sy = bb[3] - bb[2]
            if len(pos)==6:
                pos[2] = bb[4] - bb[5]

        if utils.is_sequence(sx) and utils.is_sequence(sy):
            verts = []
            for y in sy:
                for x in sx:
                    verts.append([x, y, 0])
            faces = []
            n = len(sx)
            m = len(sy)
            for j in range(m - 1):
                j1n = (j + 1) * n
                for i in range(n - 1):
                    faces.append([i + j * n, i + 1 + j * n, i + 1 + j1n, i + j1n])

            super().__init__([verts, faces], c, alpha)

        else:
            ps = vtki.new("PlaneSource")
            ps.SetResolution(resx, resy)
            ps.Update()

            t = vtki.vtkTransform()
            t.Translate(pos)
            t.Scale(sx, sy, 1)

            tf = vtki.new("TransformPolyDataFilter")
            tf.SetInputData(ps.GetOutput())
            tf.SetTransform(t)
            tf.Update()

            super().__init__(tf.GetOutput(), c, alpha)

        self.wireframe().lw(lw)
        self.properties.LightingOff()
        self.name = "Grid"


class Plane(Mesh):
    """Create a plane in space."""
    def __init__(
        self,
        pos=(0, 0, 0),
        normal=(0, 0, 1),
        s=(1, 1),
        res=(1, 1),
        edge_direction=(),
        c="gray5",
        alpha=1.0,
    ) -> None:
        """
        Create a plane of size `s=(xsize, ysize)` oriented perpendicular
        to vector `normal` so that it passes through point `pos`, optionally
        aligning an edge with `direction`.

        Arguments:
            pos : (list)
                position of the plane center
            normal : (list)
                normal vector to the plane
            s : (list)
                size of the plane along x and y
            res : (list)
                resolution of the plane along x and y
            edge_direction : (list)
                direction vector to align one edge of the plane
        """
        if isinstance(pos, vtki.vtkPolyData):
            super().__init__(pos, c, alpha)

        else:
            ps = vtki.new("PlaneSource")
            ps.SetResolution(res[0], res[1])
            tri = vtki.new("TriangleFilter")
            tri.SetInputConnection(ps.GetOutputPort())
            tri.Update()
            super().__init__(tri.GetOutput(), c, alpha)

            pos = utils.make3d(pos)
            normal = np.asarray(normal, dtype=float)
            axis = normal / np.linalg.norm(normal)

            # Calculate orientation using normal
            theta = np.arccos(axis[2])
            phi = np.arctan2(axis[1], axis[0])

            t = LinearTransform()
            t.scale([s[0], s[1], 1])

            # Rotate to align normal
            t.rotate_y(np.rad2deg(theta))
            t.rotate_z(np.rad2deg(phi))

            # Additional direction alignment
            if len(edge_direction) >= 2:
                direction = utils.make3d(edge_direction).astype(float)
                direction /= np.linalg.norm(direction)

                if s[0] <= s[1]:
                    current_direction = np.asarray([0,1,0])
                else:
                    current_direction = np.asarray([1,0,0])

                transformed_current_direction = t.transform_point(current_direction)
                n = transformed_current_direction / np.linalg.norm(transformed_current_direction)

                if np.linalg.norm(transformed_current_direction) >= 1e-6:
                    angle = np.arccos(np.dot(n, direction))
                    t.rotate(axis=axis, angle=np.rad2deg(angle))

            t.translate(pos)
            self.apply_transform(t)

        self.lighting("off")
        self.name = "Plane"
        self.variance = 0 # used by pointcloud.fit_plane()

    def clone(self, deep=True) -> Plane:
        newplane = Plane()
        if deep:
            newplane.dataset.DeepCopy(self.dataset)
        else:
            newplane.dataset.ShallowCopy(self.dataset)
        newplane.copy_properties_from(self)
        newplane.transform = self.transform.clone()
        newplane.variance = 0
        return newplane

    @property
    def normal(self) -> np.ndarray:
        pts = self.coordinates
        # this is necessary because plane can have high resolution
        # p0, p1 = pts[0], pts[1]
        # AB = p1 - p0
        # AB /= np.linalg.norm(AB)
        # for pt in pts[2:]:
        #     AC = pt - p0
        #     AC /= np.linalg.norm(AC)
        #     cosine_angle = np.dot(AB, AC)
        #     if abs(cosine_angle) < 0.99:
        #         normal = np.cross(AB, AC)
        #         return normal / np.linalg.norm(normal)
        p0, p1, p2 = pts[0], pts[1], pts[int(len(pts)/2 +0.5)]
        AB = p1 - p0
        AB /= np.linalg.norm(AB)
        AC = p2 - p0
        AC /= np.linalg.norm(AC)
        normal = np.cross(AB, AC)
        return normal / np.linalg.norm(normal)

    @property
    def center(self) -> np.ndarray:
        pts = self.coordinates
        return np.mean(pts, axis=0)

    def contains(self, points, tol=0) -> np.ndarray:
        """
        Check if each of the provided point lies on this plane.
        `points` is an array of shape (n, 3).
        """
        points = np.array(points, dtype=float)
        bounds = self.coordinates

        mask = np.isclose(np.dot(points - self.center, self.normal), 0, atol=tol)

        for i in [1, 3]:
            AB = bounds[i] - bounds[0]
            AP = points - bounds[0]
            mask_l = np.less_equal(np.dot(AP, AB), np.linalg.norm(AB))
            mask_g = np.greater_equal(np.dot(AP, AB), 0)
            mask = np.logical_and(mask, mask_l)
            mask = np.logical_and(mask, mask_g)
        return mask


class Rectangle(Mesh):
    """
    Build a rectangle in the xy plane.
    """

    def __init__(self, p1=(0, 0), p2=(1, 1), radius=None, res=12, c="gray5", alpha=1.0) -> None:
        """
        Build a rectangle in the xy plane identified by any two corner points.

        Arguments:
            p1 : (list)
                bottom-left position of the corner
            p2 : (list)
                top-right position of the corner
            radius : (float, list)
                smoothing radius of the corner in world units.
                A list can be passed with 4 individual values.
        """
        if len(p1) == 2:
            p1 = np.array([p1[0], p1[1], 0.0])
        else:
            p1 = np.array(p1, dtype=float)
        if len(p2) == 2:
            p2 = np.array([p2[0], p2[1], 0.0])
        else:
            p2 = np.array(p2, dtype=float)

        self.corner1 = p1
        self.corner2 = p2

        color = c
        smoothr = False
        risseq = False
        if utils.is_sequence(radius):
            risseq = True
            smoothr = True
            if max(radius) == 0:
                smoothr = False
        elif radius:
            smoothr = True

        if not smoothr:
            radius = None
        self.radius = radius

        if smoothr:
            r = radius
            if not risseq:
                r = [r, r, r, r]
            rd, ra, rb, rc = r

            if p1[0] > p2[0]:  # flip p1 - p2
                p1, p2 = p2, p1
            if p1[1] > p2[1]:  # flip p1y - p2y
                p1[1], p2[1] = p2[1], p1[1]

            px, py, _ = p2 - p1
            k = min(px / 2, py / 2)
            ra = min(abs(ra), k)
            rb = min(abs(rb), k)
            rc = min(abs(rc), k)
            rd = min(abs(rd), k)
            beta = np.linspace(0, 2 * np.pi, num=res * 4, endpoint=False)
            betas = np.split(beta, 4)
            rrx = np.cos(betas)
            rry = np.sin(betas)

            q1 = (rd, 0)
            # q2 = (px-ra, 0)
            q3 = (px, ra)
            # q4 = (px, py-rb)
            q5 = (px - rb, py)
            # q6 = (rc, py)
            q7 = (0, py - rc)
            # q8 = (0, rd)
            a = np.c_[rrx[3], rry[3]]*ra + [px-ra, ra]    if ra else np.array([])
            b = np.c_[rrx[0], rry[0]]*rb + [px-rb, py-rb] if rb else np.array([])
            c = np.c_[rrx[1], rry[1]]*rc + [rc, py-rc]    if rc else np.array([])
            d = np.c_[rrx[2], rry[2]]*rd + [rd, rd]       if rd else np.array([])

            pts = [q1, *a.tolist(), q3, *b.tolist(), q5, *c.tolist(), q7, *d.tolist()]
            faces = [list(range(len(pts)))]
        else:
            p1r = np.array([p2[0], p1[1], 0.0])
            p2l = np.array([p1[0], p2[1], 0.0])
            pts = ([0.0, 0.0, 0.0], p1r - p1, p2 - p1, p2l - p1)
            faces = [(0, 1, 2, 3)]

        super().__init__([pts, faces], color, alpha)
        self.pos(p1)
        self.properties.LightingOff()
        self.name = "Rectangle"
