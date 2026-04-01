#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Volumetric and 3D primitive shapes."""

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


class Box(Mesh):
    """
    Build a box of specified dimensions.
    """

    def __init__(
        self,
        pos=(0, 0, 0),
        length=1.0,
        width=1.0,
        height=1.0,
        size=(),
        c="g4",
        alpha=1.0,
    ) -> None:
        """
        Build a box of dimensions `x=length, y=width and z=height`.
        Alternatively dimensions can be defined by setting `size` keyword with a tuple.

        If `pos` is a list of 6 numbers, this will be interpreted as the bounding box:
        `[xmin,xmax, ymin,ymax, zmin,zmax]`

        Note that the shape polygonal data contains duplicated vertices. This is to allow
        each face to have its own normal, which is essential for some operations.
        Use the `clean()` method to remove duplicate points.

        Examples:
            - [aspring1.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/aspring1.py)

                ![](https://vedo.embl.es/images/simulations/50738955-7e891800-11d9-11e9-85cd-02bd4f3f13ea.gif)
        """
        src = vtki.new("CubeSource")

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        #################
        if len(pos) == 6:
            length, width, height = (
                (pos[1] - pos[0]),
                (pos[3] - pos[2]),
                (pos[5] - pos[4]),
            )
            pos = [(pos[0] + pos[1]) / 2, (pos[2] + pos[3]) / 2, (pos[4] + pos[5]) / 2]

        elif len(size) == 3:
            length, width, height = size

        src.SetXLength(length)
        src.SetYLength(width)
        src.SetZLength(height)

        src.Update()
        pd = src.GetOutput()

        tc = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        vtc = utils.numpy2vtk(tc)
        pd.GetPointData().SetTCoords(vtc)
        super().__init__(pd, c, alpha)
        self.name = "Box"
        self.pos(pos)


class Cube(Box):
    """
    Build a cube shape.

    Note that the shape polygonal data contains duplicated vertices. This is to allow
    each face to have its own normal, which is essential for some operations.
    Use the `clean()` method to remove duplicate points.
    """

    def __init__(self, pos=(0, 0, 0), side=1.0, c="g4", alpha=1.0) -> None:
        """Build a cube of size `side`."""
        super().__init__(pos, side, side, side, (), c, alpha)
        self.name = "Cube"


class TessellatedBox(Mesh):
    """
    Build a cubic `Mesh` made of quads.
    """

    def __init__(
        self, pos=(0, 0, 0), n=10, spacing=(1, 1, 1), bounds=(), c="k5", alpha=0.5
    ) -> None:
        """
        Build a cubic `Mesh` made of `n` small quads in the 3 axis directions.

        Arguments:
            pos : (list)
                position of the left bottom corner
            n : (int, list)
                number of subdivisions along each side
            spacing : (float)
                size of the side of the single quad in the 3 directions
        """
        if utils.is_sequence(n):  # slow
            img = vtki.vtkImageData()
            img.SetDimensions(n[0] + 1, n[1] + 1, n[2] + 1)
            img.SetSpacing(spacing)
            gf = vtki.new("GeometryFilter")
            gf.SetInputData(img)
            gf.Update()
            poly = gf.GetOutput()
        else:  # fast
            n -= 1
            tbs = vtki.new("TessellatedBoxSource")
            tbs.SetLevel(n)
            if len(bounds) > 0:
                tbs.SetBounds(bounds)
            else:
                tbs.SetBounds(0, n * spacing[0], 0, n * spacing[1], 0, n * spacing[2])
            tbs.QuadsOn()
            # tbs.SetOutputPointsPrecision(vtki.vtkAlgorithm.SINGLE_PRECISION)
            tbs.Update()
            poly = tbs.GetOutput()
        super().__init__(poly, c=c, alpha=alpha)
        self.pos(pos)
        self.lw(1).lighting("off")
        self.name = "TessellatedBox"


class Spring(Mesh):
    """
    Build a spring model.
    """

    def __init__(
        self,
        start_pt=(0, 0, 0),
        end_pt=(1, 0, 0),
        coils=20,
        r1=0.1,
        r2=None,
        thickness=None,
        c="gray5",
        alpha=1.0,
    ) -> None:
        """
        Build a spring of specified nr of `coils` between `start_pt` and `end_pt`.

        Arguments:
            coils : (int)
                number of coils
            r1 : (float)
                radius at start point
            r2 : (float)
                radius at end point
            thickness : (float)
                thickness of the coil section
        """
        start_pt = utils.make3d(start_pt)
        end_pt = utils.make3d(end_pt)

        diff = end_pt - start_pt
        length = np.linalg.norm(diff)
        if not length:
            return
        if not r1:
            r1 = length / 20
        trange = np.linspace(0, length, num=50 * coils)
        om = 6.283 * (coils - 0.5) / length
        if not r2:
            r2 = r1
        pts = []
        for t in trange:
            f = (length - t) / length
            rd = r1 * f + r2 * (1 - f)
            pts.append([rd * np.cos(om * t), rd * np.sin(om * t), t])

        pts = [[0, 0, 0]] + pts + [[0, 0, length]]
        diff = diff / length
        theta = np.arccos(diff[2])
        phi = np.arctan2(diff[1], diff[0])
        # Local import avoids circular dependencies during shapes package initialization.
        from .curves_core import Line

        sp = Line(pts)

        t = vtki.vtkTransform()
        t.Translate(start_pt)
        t.RotateZ(np.rad2deg(phi))
        t.RotateY(np.rad2deg(theta))

        tf = vtki.new("TransformPolyDataFilter")
        tf.SetInputData(sp.dataset)
        tf.SetTransform(t)
        tf.Update()

        tuf = vtki.new("TubeFilter")
        tuf.SetNumberOfSides(12)
        tuf.CappingOn()
        tuf.SetInputData(tf.GetOutput())
        if not thickness:
            thickness = r1 / 10
        tuf.SetRadius(thickness)
        tuf.Update()

        super().__init__(tuf.GetOutput(), c, alpha)

        self.phong().lighting("metallic")
        self.base = np.array(start_pt, dtype=float)
        self.top = np.array(end_pt, dtype=float)
        self.name = "Spring"


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

    def __init__(
        self, pos=(0, 0, 0), r=1.0, res=24, quads=False, c="r5", alpha=1.0
    ) -> None:
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
        self.pvalue = 0  # used by pointcloud.pca_ellipsoid()

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
        asp = (
            (
                ((a - b) / (a + b)) ** 2
                + ((a - c) / (a + c)) ** 2
                + ((b - c) / (b + c)) ** 2
            )
            / 3.0
            * 4.0
        )
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

        dL2 = (
            ea**2
            * (
                -8 * (a - b) ** 2 / (3 * (a + b) ** 3)
                - 8 * (a - c) ** 2 / (3 * (a + c) ** 3)
                + 4 * (2 * a - 2 * c) / (3 * (a + c) ** 2)
                + 4 * (2 * a - 2 * b) / (3 * (a + b) ** 2)
            )
            ** 2
            + eb**2
            * (
                4 * (-2 * a + 2 * b) / (3 * (a + b) ** 2)
                - 8 * (a - b) ** 2 / (3 * (a + b) ** 3)
                - 8 * (-b + c) ** 2 / (3 * (b + c) ** 3)
                + 4 * (2 * b - 2 * c) / (3 * (b + c) ** 2)
            )
            ** 2
            + ec**2
            * (
                4 * (-2 * a + 2 * c) / (3 * (a + c) ** 2)
                - 8 * (a - c) ** 2 / (3 * (a + c) ** 3)
                + 4 * (-2 * b + 2 * c) / (3 * (b + c) ** 2)
                - 8 * (-b + c) ** 2 / (3 * (b + c) ** 3)
            )
            ** 2
        )
        err = np.sqrt(dL2)
        self.va_error = ea
        self.vb_error = eb
        self.vc_error = ec
        return err


class Cylinder(Mesh):
    """
    Build a cylinder of specified height and radius.
    """

    def __init__(
        self,
        pos=(0, 0, 0),
        r=1.0,
        height=2.0,
        axis=(0, 0, 1),
        cap=True,
        res=24,
        c="teal3",
        alpha=1.0,
    ) -> None:
        """
        Build a cylinder of specified height and radius `r`, centered at `pos`.

        If `pos` is a list of 2 points, e.g. `pos=[v1, v2]`, build a cylinder with base
        centered at `v1` and top at `v2`.

        Arguments:
            cap : (bool)
                enable/disable the caps of the cylinder
            res : (int)
                resolution of the cylinder sides

        ![](https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestCylinder.png)
        """
        if utils.is_sequence(pos[0]):  # assume user is passing pos=[base, top]
            base = np.array(pos[0], dtype=float)
            top = np.array(pos[1], dtype=float)
            pos = (base + top) / 2
            height = np.linalg.norm(top - base)
            axis = top - base
            axis = utils.versor(axis)
        else:
            axis = utils.versor(axis)
            base = pos - axis * height / 2
            top = pos + axis * height / 2

        cyl = vtki.new("CylinderSource")
        cyl.SetResolution(res)
        cyl.SetRadius(r)
        cyl.SetHeight(height)
        cyl.SetCapping(cap)
        cyl.Update()

        theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])
        t = vtki.vtkTransform()
        t.PostMultiply()
        t.RotateX(90)  # put it along Z
        t.RotateY(np.rad2deg(theta))
        t.RotateZ(np.rad2deg(phi))
        t.Translate(pos)

        tf = vtki.new("TransformPolyDataFilter")
        tf.SetInputData(cyl.GetOutput())
        tf.SetTransform(t)
        tf.Update()

        super().__init__(tf.GetOutput(), c, alpha)

        self.phong()
        self.base = base
        self.top = top
        self.transform = LinearTransform().translate(pos)
        self.name = "Cylinder"


class Cone(Mesh):
    """Build a cone of specified radius and height."""

    def __init__(
        self,
        pos=(0, 0, 0),
        r=1.0,
        height=3.0,
        axis=(0, 0, 1),
        res=48,
        c="green3",
        alpha=1.0,
    ) -> None:
        """Build a cone of specified radius `r` and `height`, centered at `pos`."""
        con = vtki.new("ConeSource")
        con.SetResolution(res)
        con.SetRadius(r)
        con.SetHeight(height)
        con.SetDirection(axis)
        con.Update()
        super().__init__(con.GetOutput(), c, alpha)
        self.phong()
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.pos(pos)
        v = utils.versor(axis) * height / 2
        self.base = pos - v
        self.top = pos + v
        self.name = "Cone"


class Pyramid(Cone):
    """Build a pyramidal shape."""

    def __init__(
        self, pos=(0, 0, 0), s=1.0, height=1.0, axis=(0, 0, 1), c="green3", alpha=1
    ) -> None:
        """Build a pyramid of specified base size `s` and `height`, centered at `pos`."""
        super().__init__(pos, s, height, axis, 4, c, alpha)
        self.name = "Pyramid"


class Torus(Mesh):
    """
    Build a toroidal shape.
    """

    def __init__(
        self, pos=(0, 0, 0), r1=1.0, r2=0.2, res=36, quads=False, c="yellow3", alpha=1.0
    ) -> None:
        """
        Build a torus of specified outer radius `r1` internal radius `r2`, centered at `pos`.
        If `quad=True` a quad-mesh is generated.
        """
        if utils.is_sequence(res):
            res_u, res_v = res
        else:
            res_u, res_v = 3 * res, res

        if quads:
            # https://github.com/marcomusy/vedo/issues/710

            n = res_v
            m = res_u

            theta = np.linspace(0, 2.0 * np.pi, n)
            phi = np.linspace(0, 2.0 * np.pi, m)
            theta, phi = np.meshgrid(theta, phi)
            t = r1 + r2 * np.cos(theta)
            x = t * np.cos(phi)
            y = t * np.sin(phi)
            z = r2 * np.sin(theta)
            pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

            faces = []
            for j in range(m - 1):
                j1n = (j + 1) * n
                for i in range(n - 1):
                    faces.append([i + j * n, i + 1 + j * n, i + 1 + j1n, i + j1n])

            super().__init__([pts, faces], c, alpha)

        else:
            rs = vtki.new("ParametricTorus")
            rs.SetRingRadius(r1)
            rs.SetCrossSectionRadius(r2)
            pfs = vtki.new("ParametricFunctionSource")
            pfs.SetParametricFunction(rs)
            pfs.SetUResolution(res_u)
            pfs.SetVResolution(res_v)
            pfs.Update()

            super().__init__(pfs.GetOutput(), c, alpha)

        self.phong()
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.pos(pos)
        self.name = "Torus"


class Paraboloid(Mesh):
    """
    Build a paraboloid.
    """

    def __init__(self, pos=(0, 0, 0), height=1.0, res=50, c="cyan5", alpha=1.0) -> None:
        """
        Build a paraboloid of specified height and radius `r`, centered at `pos`.

        Full volumetric expression is:
            `F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`

        ![](https://user-images.githubusercontent.com/32848391/51211547-260ef480-1916-11e9-95f6-4a677e37e355.png)
        """
        quadric = vtki.new("Quadric")
        quadric.SetCoefficients(1, 1, 0, 0, 0, 0, 0, 0, height / 4, 0)
        # F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2
        #         + a3*x*y + a4*y*z + a5*x*z
        #         + a6*x   + a7*y   + a8*z  +a9
        sample = vtki.new("SampleFunction")
        sample.SetSampleDimensions(res, res, res)
        sample.SetImplicitFunction(quadric)

        contours = vtki.new("ContourFilter")
        contours.SetInputConnection(sample.GetOutputPort())
        contours.GenerateValues(1, 0.01, 0.01)
        contours.Update()

        super().__init__(contours.GetOutput(), c, alpha)
        self.compute_normals().phong()
        self.mapper.ScalarVisibilityOff()
        self.pos(pos)
        self.name = "Paraboloid"


class Hyperboloid(Mesh):
    """
    Build a hyperboloid.
    """

    def __init__(
        self, pos=(0, 0, 0), a2=1.0, value=0.5, res=100, c="pink4", alpha=1.0
    ) -> None:
        """
        Build a hyperboloid of specified aperture `a2` and `height`, centered at `pos`.

        Full volumetric expression is:
            `F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`
        """
        q = vtki.new("Quadric")
        q.SetCoefficients(2, 2, -1 / a2, 0, 0, 0, 0, 0, 0, 0)
        # F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2
        #         + a3*x*y + a4*y*z + a5*x*z
        #         + a6*x   + a7*y   + a8*z  +a9
        sample = vtki.new("SampleFunction")
        sample.SetSampleDimensions(res, res, res)
        sample.SetImplicitFunction(q)

        contours = vtki.new("ContourFilter")
        contours.SetInputConnection(sample.GetOutputPort())
        contours.GenerateValues(1, value, value)
        contours.Update()

        super().__init__(contours.GetOutput(), c, alpha)
        self.compute_normals().phong()
        self.mapper.ScalarVisibilityOff()
        self.pos(pos)
        self.name = "Hyperboloid"
