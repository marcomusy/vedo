#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PointCutMixin extracted from pointcloud core."""

from typing import Union, List
from typing_extensions import Self

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils
from vedo.transformations import LinearTransform, NonLinearTransform
from ._proxy import Points

class PointCutMixin:
    def cut_with_plane(
            self,
            origin=(0, 0, 0),
            normal=(1, 0, 0),
            invert=False,
            # generate_ids=False,
    ) -> Self:
        """
        Cut the mesh with the plane defined by a point and a normal.

        Arguments:
            origin : (array)
                the cutting plane goes through this point
            normal : (array)
                normal of the cutting plane
            invert : (bool)
                select which side of the plane to keep

        Example:
            ```python
            from vedo import Cube
            cube = Cube().cut_with_plane(normal=(1,1,1))
            cube.back_color('pink').show().close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_plane_cube.png)

        Examples:
            - [trail.py](https://github.com/marcomusy/vedo/blob/master/examples/simulations/trail.py)

                ![](https://vedo.embl.es/images/simulations/trail.gif)

        Check out also:
            `cut_with_box()`, `cut_with_cylinder()`, `cut_with_sphere()`.
        """
        s = str(normal)
        if "x" in s:
            normal = (1, 0, 0)
            if "-" in s:
                normal = -np.array(normal)
        elif "y" in s:
            normal = (0, 1, 0)
            if "-" in s:
                normal = -np.array(normal)
        elif "z" in s:
            normal = (0, 0, 1)
            if "-" in s:
                normal = -np.array(normal)
        plane = vtki.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(plane)
        clipper.GenerateClippedOutputOff()
        clipper.SetGenerateClipScalars(0)
        clipper.SetInsideOut(invert)
        clipper.SetValue(0)
        clipper.Update()

        # if generate_ids:
        #     saved_scalars = None # otherwise the scalars are lost
        #     if self.dataset.GetPointData().GetScalars():
        #         saved_scalars = self.dataset.GetPointData().GetScalars()
        #     varr = clipper.GetOutput().GetPointData().GetScalars()
        #     if varr.GetName() is None:
        #         varr.SetName("DistanceToCut")
        #     arr = utils.vtk2numpy(varr)
        #     # array of original ids
        #     ids = np.arange(arr.shape[0]).astype(int)
        #     ids[arr == 0] = -1
        #     ids_arr = utils.numpy2vtk(ids, dtype=int)
        #     ids_arr.SetName("OriginalIds")
        #     clipper.GetOutput().GetPointData().AddArray(ids_arr)
        #     if saved_scalars:
        #         clipper.GetOutput().GetPointData().AddArray(saved_scalars)

        self._update(clipper.GetOutput())
        self.pipeline = utils.OperationNode("cut_with_plane", parents=[self])
        return self

    def cut_with_planes(self, origins, normals, invert=False) -> Self:
        """
        Cut the mesh with a convex set of planes defined by points and normals.

        Arguments:
            origins : (array)
                each cutting plane goes through this point
            normals : (array)
                normal of each of the cutting planes
            invert : (bool)
                if True, cut outside instead of inside

        Check out also:
            `cut_with_box()`, `cut_with_cylinder()`, `cut_with_sphere()`
        """

        vpoints = vtki.vtkPoints()
        for p in utils.make3d(origins):
            vpoints.InsertNextPoint(p)
        normals = utils.make3d(normals)

        planes = vtki.vtkPlanes()
        planes.SetPoints(vpoints)
        planes.SetNormals(utils.numpy2vtk(normals, dtype=float))

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetInsideOut(invert)
        clipper.SetClipFunction(planes)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()

        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_planes", parents=[self])
        return self

    def cut_with_box(self, bounds, invert=False) -> Self:
        """
        Cut the current mesh with a box or a set of boxes.
        This is much faster than `cut_with_mesh()`.

        Input `bounds` can be either:
        - a Mesh or Points object
        - a list of 6 number representing a bounding box `[xmin,xmax, ymin,ymax, zmin,zmax]`
        - a list of bounding boxes like the above: `[[xmin1,...], [xmin2,...], ...]`

        Example:
            ```python
            from vedo import Sphere, Cube, show
            mesh = Sphere(r=1, res=50)
            box  = Cube(side=1.5).wireframe()
            mesh.cut_with_box(box)
            show(mesh, box, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_box_cube.png)

        Check out also:
            `cut_with_line()`, `cut_with_plane()`, `cut_with_cylinder()`
        """
        if isinstance(bounds, Points):
            bounds = bounds.bounds()

        box = vtki.new("Box")
        if utils.is_sequence(bounds[0]):
            for bs in bounds:
                box.AddBounds(bs)
        else:
            box.SetBounds(bounds)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(box)
        clipper.SetInsideOut(not invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_box", parents=[self])
        return self

    def cut_with_line(self, points, invert=False, closed=True) -> Self:
        """
        Cut the current mesh with a line vertically in the z-axis direction like a cookie cutter.
        The polyline is defined by a set of points (z-coordinates are ignored).
        This is much faster than `cut_with_mesh()`.

        Check out also:
            `cut_with_box()`, `cut_with_plane()`, `cut_with_sphere()`
        """
        pplane = vtki.new("PolyPlane")
        if isinstance(points, Points):
            points = points.coordinates.tolist()

        if closed:
            if isinstance(points, np.ndarray):
                points = points.tolist()
            points.append(points[0])

        vpoints = vtki.vtkPoints()
        for p in points:
            if len(p) == 2:
                p = [p[0], p[1], 0.0]
            vpoints.InsertNextPoint(p)

        n = len(points)
        polyline = vtki.new("PolyLine")
        polyline.Initialize(n, vpoints)
        polyline.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            polyline.GetPointIds().SetId(i, i)
        pplane.SetPolyLine(polyline)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(pplane)
        clipper.SetInsideOut(invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_line", parents=[self])
        return self

    def cut_with_cookiecutter(self, lines) -> Self:
        """
        Cut the current mesh with a single line or a set of lines.

        Input `lines` can be either:
        - a `Mesh` or `Points` object
        - a list of 3D points: `[(x1,y1,z1), (x2,y2,z2), ...]`
        - a list of 2D points: `[(x1,y1), (x2,y2), ...]`

        Example:
            ```python
            from vedo import *
            grid = Mesh(dataurl + "dolfin_fine.vtk")
            grid.compute_quality().cmap("Greens")
            pols = merge(
                Polygon(nsides=10, r=0.3).pos(0.7, 0.3),
                Polygon(nsides=10, r=0.2).pos(0.3, 0.7),
            )
            lines = pols.boundaries()
            cgrid = grid.clone().cut_with_cookiecutter(lines)
            grid.alpha(0.1).wireframe()
            show(grid, cgrid, lines, axes=8, bg='blackboard').close()
            ```
            ![](https://vedo.embl.es/images/feats/cookiecutter.png)

        Check out also:
            `cut_with_line()` and `cut_with_point_loop()`

        Note:
            In case of a warning message like:
                "Mesh and trim loop point data attributes are different"
            consider interpolating the mesh point data to the loop points,
            Eg. (in the above example):
            ```python
            lines = pols.boundaries().interpolate_data_from(grid, n=2)
            ```

        Note:
            trying to invert the selection by reversing the loop order
            will have no effect in this method, hence it does not have
            the `invert` option.
        """
        if utils.is_sequence(lines):
            lines = utils.make3d(lines)
            iline = list(range(len(lines))) + [0]
            poly = utils.buildPolyData(lines, lines=[iline])
        else:
            poly = lines.dataset

        # if invert: # not working
        #     rev = vtki.new("ReverseSense")
        #     rev.ReverseCellsOn()
        #     rev.SetInputData(poly)
        #     rev.Update()
        #     poly = rev.GetOutput()

        # Build loops from the polyline
        build_loops = vtki.new("ContourLoopExtraction")
        build_loops.SetGlobalWarningDisplay(0)
        build_loops.SetInputData(poly)
        build_loops.Update()
        boundary_poly = build_loops.GetOutput()

        ccut = vtki.new("CookieCutter")
        ccut.SetInputData(self.dataset)
        ccut.SetLoopsData(boundary_poly)
        ccut.SetPointInterpolationToMeshEdges()
        # ccut.SetPointInterpolationToLoopEdges()
        ccut.PassCellDataOn()
        ccut.PassPointDataOn()
        ccut.Update()
        self._update(ccut.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_cookiecutter", parents=[self])
        return self

    def cut_with_cylinder(self, center=(0, 0, 0), axis=(0, 0, 1), r=1, invert=False) -> Self:
        """
        Cut the current mesh with an infinite cylinder.
        This is much faster than `cut_with_mesh()`.

        Arguments:
            center : (array)
                the center of the cylinder
            normal : (array)
                direction of the cylinder axis
            r : (float)
                radius of the cylinder

        Example:
            ```python
            from vedo import Disc, show
            disc = Disc(r1=1, r2=1.2)
            mesh = disc.extrude(3, res=50).linewidth(1)
            mesh.cut_with_cylinder([0,0,2], r=0.4, axis='y', invert=True)
            show(mesh, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_cylinder.png)

        Examples:
            - [optics_main1.py](https://github.com/marcomusy/vedo/blob/master/examples/simulations/optics_main1.py)

        Check out also:
            `cut_with_box()`, `cut_with_plane()`, `cut_with_sphere()`
        """
        s = str(axis)
        if "x" in s:
            axis = (1, 0, 0)
        elif "y" in s:
            axis = (0, 1, 0)
        elif "z" in s:
            axis = (0, 0, 1)
        cyl = vtki.new("Cylinder")
        cyl.SetCenter(center)
        cyl.SetAxis(axis[0], axis[1], axis[2])
        cyl.SetRadius(r)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(cyl)
        clipper.SetInsideOut(not invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_cylinder", parents=[self])
        return self

    def cut_with_sphere(self, center=(0, 0, 0), r=1.0, invert=False) -> Self:
        """
        Cut the current mesh with an sphere.
        This is much faster than `cut_with_mesh()`.

        Arguments:
            center : (array)
                the center of the sphere
            r : (float)
                radius of the sphere

        Example:
            ```python
            from vedo import Disc, show
            disc = Disc(r1=1, r2=1.2)
            mesh = disc.extrude(3, res=50).linewidth(1)
            mesh.cut_with_sphere([1,-0.7,2], r=1.5, invert=True)
            show(mesh, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_sphere.png)

        Check out also:
            `cut_with_box()`, `cut_with_plane()`, `cut_with_cylinder()`
        """
        sph = vtki.new("Sphere")
        sph.SetCenter(center)
        sph.SetRadius(r)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(sph)
        clipper.SetInsideOut(not invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())
        self.pipeline = utils.OperationNode("cut_with_sphere", parents=[self])
        return self

    def cut_with_mesh(self, mesh, invert=False, keep=False) -> Union[Self, "vedo.Assembly"]:
        """
        Cut an `Mesh` mesh with another `Mesh`.

        Use `invert` to invert the selection.

        Use `keep` to keep the cutoff part, in this case an `Assembly` is returned:
        the "cut" object and the "discarded" part of the original object.
        You can access both via `assembly.unpack()` method.

        Example:
        ```python
        from vedo import *
        arr = np.random.randn(100000, 3)/2
        pts = Points(arr).c('red3').pos(5,0,0)
        cube = Cube().pos(4,0.5,0)
        assem = pts.cut_with_mesh(cube, keep=True)
        show(assem.unpack(), axes=1).close()
        ```
        ![](https://vedo.embl.es/images/feats/cut_with_mesh.png)

       Check out also:
            `cut_with_box()`, `cut_with_plane()`, `cut_with_cylinder()`
       """
        polymesh = mesh.dataset
        poly = self.dataset

        # Create an array to hold distance information
        signed_distances = vtki.vtkFloatArray()
        signed_distances.SetNumberOfComponents(1)
        signed_distances.SetName("SignedDistances")

        # implicit function that will be used to slice the mesh
        ippd = vtki.new("ImplicitPolyDataDistance")
        ippd.SetInput(polymesh)

        # Evaluate the signed distance function at all of the grid points
        for pointId in range(poly.GetNumberOfPoints()):
            p = poly.GetPoint(pointId)
            signed_distance = ippd.EvaluateFunction(p)
            signed_distances.InsertNextValue(signed_distance)

        currentscals = poly.GetPointData().GetScalars()
        if currentscals:
            currentscals = currentscals.GetName()

        poly.GetPointData().AddArray(signed_distances)
        poly.GetPointData().SetActiveScalars("SignedDistances")

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(poly)
        clipper.SetInsideOut(not invert)
        clipper.SetGenerateClippedOutput(keep)
        clipper.SetValue(0.0)
        clipper.Update()
        cpoly = clipper.GetOutput()

        if keep:
            kpoly = clipper.GetOutput(1)

        vis = False
        if currentscals:
            cpoly.GetPointData().SetActiveScalars(currentscals)
            vis = self.mapper.GetScalarVisibility()

        self._update(cpoly)

        self.pointdata.remove("SignedDistances")
        self.mapper.SetScalarVisibility(vis)
        if keep:
            if isinstance(self, vedo.Mesh):
                cutoff = vedo.Mesh(kpoly)
            else:
                cutoff = vedo.Points(kpoly)
            # cutoff = self.__class__(kpoly) # this does not work properly
            cutoff.properties = vtki.vtkProperty()
            cutoff.properties.DeepCopy(self.properties)
            cutoff.actor.SetProperty(cutoff.properties)
            cutoff.c("k5").alpha(0.2)
            return vedo.Assembly([self, cutoff])

        self.pipeline = utils.OperationNode("cut_with_mesh", parents=[self, mesh])
        return self

    def cut_with_point_loop(
        self, points, invert=False, on="points", include_boundary=False
    ) -> Self:
        """
        Cut an `Mesh` object with a set of points forming a closed loop.

        Arguments:
            invert : (bool)
                invert selection (inside-out)
            on : (str)
                if 'cells' will extract the whole cells lying inside (or outside) the point loop
            include_boundary : (bool)
                include cells lying exactly on the boundary line. Only relevant on 'cells' mode

        Examples:
            - [cut_with_points1.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/cut_with_points1.py)

                ![](https://vedo.embl.es/images/advanced/cutWithPoints1.png)

            - [cut_with_points2.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/cut_with_points2.py)

                ![](https://vedo.embl.es/images/advanced/cutWithPoints2.png)
        """
        if isinstance(points, Points):
            parents = [points]
            vpts = points.dataset.GetPoints()
            points = points.coordinates
        else:
            parents = [self]
            vpts = vtki.vtkPoints()
            points = utils.make3d(points)
            for p in points:
                vpts.InsertNextPoint(p)

        if "cell" in on:
            ippd = vtki.new("ImplicitSelectionLoop")
            ippd.SetLoop(vpts)
            ippd.AutomaticNormalGenerationOn()
            clipper = vtki.new("ExtractPolyDataGeometry")
            clipper.SetInputData(self.dataset)
            clipper.SetImplicitFunction(ippd)
            clipper.SetExtractInside(not invert)
            clipper.SetExtractBoundaryCells(include_boundary)
        else:
            spol = vtki.new("SelectPolyData")
            spol.SetLoop(vpts)
            spol.GenerateSelectionScalarsOn()
            spol.GenerateUnselectedOutputOff()
            spol.SetInputData(self.dataset)
            spol.Update()
            clipper = vtki.new("ClipPolyData")
            clipper.SetInputData(spol.GetOutput())
            clipper.SetInsideOut(not invert)
            clipper.SetValue(0.0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_pointloop", parents=parents)
        return self

    def cut_with_scalar(self, value: float, name="", invert=False) -> Self:
        """
        Cut a mesh or point cloud with some input scalar point-data.

        Arguments:
            value : (float)
                cutting value
            name : (str)
                array name of the scalars to be used
            invert : (bool)
                flip selection

        Example:
            ```python
            from vedo import *
            s = Sphere().lw(1)
            pts = s.points
            scalars = np.sin(3*pts[:,2]) + pts[:,0]
            s.pointdata["somevalues"] = scalars
            s.cut_with_scalar(0.3)
            s.cmap("Spectral", "somevalues").add_scalarbar()
            s.show(axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_scalars.png)
        """
        if name:
            self.pointdata.select(name)
        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetValue(value)
        clipper.GenerateClippedOutputOff()
        clipper.SetInsideOut(not invert)
        clipper.Update()
        self._update(clipper.GetOutput())
        self.pipeline = utils.OperationNode("cut_with_scalar", parents=[self])
        return self

    def crop(self,
             top=None, bottom=None, right=None, left=None, front=None, back=None,
             bounds=()) -> Self:
        """
        Crop an `Mesh` object.

        Arguments:
            top : (float)
                fraction to crop from the top plane (positive z)
            bottom : (float)
                fraction to crop from the bottom plane (negative z)
            front : (float)
                fraction to crop from the front plane (positive y)
            back : (float)
                fraction to crop from the back plane (negative y)
            right : (float)
                fraction to crop from the right plane (positive x)
            left : (float)
                fraction to crop from the left plane (negative x)
            bounds : (list)
                bounding box of the crop region as `[x0,x1, y0,y1, z0,z1]`

        Example:
            ```python
            from vedo import Sphere
            Sphere().crop(right=0.3, left=0.1).show()
            ```
            ![](https://user-images.githubusercontent.com/32848391/57081955-0ef1e800-6cf6-11e9-99de-b45220939bc9.png)
        """
        if len(bounds) == 0:
            pos = np.array(self.pos())
            x0, x1, y0, y1, z0, z1 = self.bounds()
            x0, y0, z0 = [x0, y0, z0] - pos
            x1, y1, z1 = [x1, y1, z1] - pos

            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            if top:
                z1 = z1 - top * dz
            if bottom:
                z0 = z0 + bottom * dz
            if front:
                y1 = y1 - front * dy
            if back:
                y0 = y0 + back * dy
            if right:
                x1 = x1 - right * dx
            if left:
                x0 = x0 + left * dx
            bounds = (x0, x1, y0, y1, z0, z1)

        cu = vtki.new("Box")
        cu.SetBounds(bounds)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(cu)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode(
            "crop", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

