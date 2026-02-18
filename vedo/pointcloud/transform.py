#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""PointTransformMixin extracted from pointcloud core."""

from typing_extensions import Self

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils
from vedo.core.transformations import LinearTransform, NonLinearTransform

class PointTransformMixin:
    def align_to(self, target, iters=100, rigid=False, invert=False, use_centroids=False) -> Self:
        """
        Aligned to target mesh through the `Iterative Closest Point` algorithm.

        The core of the algorithm is to match each vertex in one surface with
        the closest surface point on the other, then apply the transformation
        that modify one surface to best match the other (in the least-square sense).

        Arguments:
            rigid : (bool)
                if True do not allow scaling
            invert : (bool)
                if True start by aligning the target to the source but
                invert the transformation finally. Useful when the target is smaller
                than the source.
            use_centroids : (bool)
                start by matching the centroids of the two objects.

        Examples:
            - [align1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align1.py)

                ![](https://vedo.embl.es/images/basic/align1.png)

            - [align2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align2.py)

                ![](https://vedo.embl.es/images/basic/align2.png)
        """
        icp = vtki.new("IterativeClosestPointTransform")
        icp.SetSource(self.dataset)
        icp.SetTarget(target.dataset)
        if invert:
            icp.Inverse()
        icp.SetMaximumNumberOfIterations(iters)
        if rigid:
            icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetStartByMatchingCentroids(use_centroids)
        icp.Update()

        self.apply_transform(icp.GetMatrix())

        self.pipeline = utils.OperationNode(
            "align_to", parents=[self, target], comment=f"rigid = {rigid}"
        )
        return self

    def align_to_bounding_box(self, msh, rigid=False) -> Self:
        """
        Align the current object's bounding box to the bounding box
        of the input object.

        Use `rigid=True` to disable scaling.

        Example:
            [align6.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align6.py)
        """
        lmt = vtki.vtkLandmarkTransform()
        ss = vtki.vtkPoints()
        xss0, xss1, yss0, yss1, zss0, zss1 = self.bounds()
        for p in [
            [xss0, yss0, zss0],
            [xss1, yss0, zss0],
            [xss1, yss1, zss0],
            [xss0, yss1, zss0],
            [xss0, yss0, zss1],
            [xss1, yss0, zss1],
            [xss1, yss1, zss1],
            [xss0, yss1, zss1],
        ]:
            ss.InsertNextPoint(p)
        st = vtki.vtkPoints()
        xst0, xst1, yst0, yst1, zst0, zst1 = msh.bounds()
        for p in [
            [xst0, yst0, zst0],
            [xst1, yst0, zst0],
            [xst1, yst1, zst0],
            [xst0, yst1, zst0],
            [xst0, yst0, zst1],
            [xst1, yst0, zst1],
            [xst1, yst1, zst1],
            [xst0, yst1, zst1],
        ]:
            st.InsertNextPoint(p)

        lmt.SetSourceLandmarks(ss)
        lmt.SetTargetLandmarks(st)
        lmt.SetModeToAffine()
        if rigid:
            lmt.SetModeToRigidBody()
        lmt.Update()

        LT = LinearTransform(lmt)
        self.apply_transform(LT)
        return self

    def align_with_landmarks(
        self,
        source_landmarks,
        target_landmarks,
        rigid=False,
        affine=False,
        least_squares=False,
    ) -> Self:
        """
        Transform mesh orientation and position based on a set of landmarks points.
        The algorithm finds the best matching of source points to target points
        in the mean least square sense, in one single step.

        If `affine` is True the x, y and z axes can scale independently but stay collinear.
        With least_squares they can vary orientation.

        Examples:
            - [align5.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align5.py)

                ![](https://vedo.embl.es/images/basic/align5.png)
        """

        if utils.is_sequence(source_landmarks):
            ss = vtki.vtkPoints()
            for p in source_landmarks:
                ss.InsertNextPoint(p)
        else:
            ss = source_landmarks.dataset.GetPoints()
            if least_squares:
                source_landmarks = source_landmarks.coordinates

        if utils.is_sequence(target_landmarks):
            st = vtki.vtkPoints()
            for p in target_landmarks:
                st.InsertNextPoint(p)
        else:
            st = target_landmarks.GetPoints()
            if least_squares:
                target_landmarks = target_landmarks.coordinates

        if ss.GetNumberOfPoints() != st.GetNumberOfPoints():
            n1 = ss.GetNumberOfPoints()
            n2 = st.GetNumberOfPoints()
            vedo.logger.error(f"source and target have different nr of points {n1} vs {n2}")
            raise RuntimeError()

        if int(rigid) + int(affine) + int(least_squares) > 1:
            vedo.logger.error(
                "only one of rigid, affine, least_squares can be True at a time"
            )
            raise RuntimeError()

        lmt = vtki.vtkLandmarkTransform()
        lmt.SetSourceLandmarks(ss)
        lmt.SetTargetLandmarks(st)
        lmt.SetModeToSimilarity()

        if rigid:
            lmt.SetModeToRigidBody()
            lmt.Update()

        elif affine:
            lmt.SetModeToAffine()
            lmt.Update()

        elif least_squares:
            cms = source_landmarks.mean(axis=0)
            cmt = target_landmarks.mean(axis=0)
            m = np.linalg.lstsq(source_landmarks - cms, target_landmarks - cmt, rcond=None)[0]
            M = vtki.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    M.SetElement(j, i, m[i][j])
            lmt = vtki.vtkTransform()
            lmt.Translate(cmt)
            lmt.Concatenate(M)
            lmt.Translate(-cms)

        else:
            lmt.Update()

        self.apply_transform(lmt)
        self.pipeline = utils.OperationNode("transform_with_landmarks", parents=[self])
        return self

    def normalize(self) -> Self:
        """Scale average size to unit. The scaling is performed around the center of mass."""
        coords = self.coordinates
        if not coords.shape[0]:
            return self
        cm = np.mean(coords, axis=0)
        pts = coords - cm
        xyz2 = np.sum(pts * pts, axis=0) / len(pts)
        scale = 1 / np.sqrt(np.sum(xyz2))
        self.scale(scale, origin=cm)
        self.pipeline = utils.OperationNode("normalize", parents=[self])
        return self

    def mirror(self, axis="x", origin=True) -> Self:
        """
        Mirror reflect along one of the cartesian axes

        Arguments:
            axis : (str)
                axis to use for mirroring, must be set to `x, y, z`.
                Or any combination of those.
            origin : (list)
                use this point as the origin of the mirroring transformation.

        Examples:
            - [mirror.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mirror.py)

                ![](https://vedo.embl.es/images/basic/mirror.png)
        """
        sx, sy, sz = 1, 1, 1
        if "x" in axis.lower(): sx = -1
        if "y" in axis.lower(): sy = -1
        if "z" in axis.lower(): sz = -1

        self.scale([sx, sy, sz], origin=origin)

        self.pipeline = utils.OperationNode(
            "mirror", comment=f"axis = {axis}", parents=[self])

        if sx * sy * sz < 0:
            if hasattr(self, "reverse"):
                self.reverse()
        return self

    def flip_normals(self) -> Self:
        """Flip all normals orientation."""
        rs = vtki.new("ReverseSense")
        rs.SetInputData(self.dataset)
        rs.ReverseCellsOff()
        rs.ReverseNormalsOn()
        rs.Update()
        self._update(rs.GetOutput())
        self.pipeline = utils.OperationNode("flip_normals", parents=[self])
        return self

    def add_gaussian_noise(self, sigma=1.0) -> Self:
        """
        Add gaussian noise to point positions.
        An extra array is added named "GaussianNoise" with the displacements.

        Arguments:
            sigma : (float)
                nr. of standard deviations, expressed in percent of the diagonal size of mesh.
                Can also be a list `[sigma_x, sigma_y, sigma_z]`.

        Example:
            ```python
            from vedo import Sphere
            Sphere().add_gaussian_noise(1.0).point_size(8).show().close()
            ```
        """
        sz = self.diagonal_size()
        pts = self.coordinates
        n = len(pts)
        ns = (np.random.randn(n, 3) * sigma) * (sz / 100)
        vpts = vtki.vtkPoints()
        vpts.SetNumberOfPoints(n)
        vpts.SetData(utils.numpy2vtk(pts + ns, dtype=np.float32))
        self.dataset.SetPoints(vpts)
        self.dataset.GetPoints().Modified()
        self.pointdata["GaussianNoise"] = -ns
        self.pipeline = utils.OperationNode(
            "gaussian_noise", parents=[self], shape="egg", comment=f"sigma = {sigma}"
        )
        return self

    def project_on_plane(self, plane="z", point=None, direction=None) -> Self:
        """
        Project the mesh on one of the Cartesian planes.

        Arguments:
            plane : (str, Plane)
                if plane is `str`, plane can be one of ['x', 'y', 'z'],
                represents x-plane, y-plane and z-plane, respectively.
                Otherwise, plane should be an instance of `vedo.shapes.Plane`.
            point : (float, array)
                if plane is `str`, point should be a float represents the intercept.
                Otherwise, point is the camera point of perspective projection
            direction : (array)
                direction of oblique projection

        Note:
            Parameters `point` and `direction` are only used if the given plane
            is an instance of `vedo.shapes.Plane`. And one of these two params
            should be left as `None` to specify the projection type.

        Example:
            ```python
            s.project_on_plane(plane='z') # project to z-plane
            plane = Plane(pos=(4, 8, -4), normal=(-1, 0, 1), s=(5,5))
            s.project_on_plane(plane=plane)                       # orthogonal projection
            s.project_on_plane(plane=plane, point=(6, 6, 6))      # perspective projection
            s.project_on_plane(plane=plane, direction=(1, 2, -1)) # oblique projection
            ```

        Examples:
            - [silhouette2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/silhouette2.py)

                ![](https://vedo.embl.es/images/basic/silhouette2.png)
        """
        coords = self.coordinates

        if plane == "x":
            coords[:, 0] = self.transform.position[0]
            intercept = self.xbounds()[0] if point is None else point
            self.x(intercept)
        elif plane == "y":
            coords[:, 1] = self.transform.position[1]
            intercept = self.ybounds()[0] if point is None else point
            self.y(intercept)
        elif plane == "z":
            coords[:, 2] = self.transform.position[2]
            intercept = self.zbounds()[0] if point is None else point
            self.z(intercept)

        elif isinstance(plane, vedo.shapes.Plane):
            normal = plane.normal / np.linalg.norm(plane.normal)
            pl = np.hstack((normal, -np.dot(plane.pos(), normal))).reshape(4, 1)
            if direction is None and point is None:
                # orthogonal projection
                pt = np.hstack((normal, [0])).reshape(4, 1)
                # proj_mat = pt.T @ pl * np.eye(4) - pt @ pl.T # python3 only
                proj_mat = np.matmul(pt.T, pl) * np.eye(4) - np.matmul(pt, pl.T)

            elif direction is None:
                # perspective projection
                pt = np.hstack((np.array(point), [1])).reshape(4, 1)
                # proj_mat = pt.T @ pl * np.eye(4) - pt @ pl.T
                proj_mat = np.matmul(pt.T, pl) * np.eye(4) - np.matmul(pt, pl.T)

            elif point is None:
                # oblique projection
                pt = np.hstack((np.array(direction), [0])).reshape(4, 1)
                # proj_mat = pt.T @ pl * np.eye(4) - pt @ pl.T
                proj_mat = np.matmul(pt.T, pl) * np.eye(4) - np.matmul(pt, pl.T)

            coords = np.concatenate([coords, np.ones((coords.shape[:-1] + (1,)))], axis=-1)
            # coords = coords @ proj_mat.T
            coords = np.matmul(coords, proj_mat.T)
            coords = coords[:, :3] / coords[:, 3:]

        else:
            vedo.logger.error(f"unknown plane {plane}")
            raise RuntimeError()

        self.alpha(0.1)
        self.coordinates = coords
        return self

    def warp(self, source, target, sigma=1.0, mode="3d") -> Self:
        """
        "Thin Plate Spline" transformations describe a nonlinear warp transform defined by a set
        of source and target landmarks. Any point on the mesh close to a source landmark will
        be moved to a place close to the corresponding target landmark.
        The points in between are interpolated smoothly using
        Bookstein's Thin Plate Spline algorithm.

        Transformation object can be accessed with `mesh.transform`.

        Arguments:
            sigma : (float)
                specify the 'stiffness' of the spline.
            mode : (str)
                set the basis function to either abs(R) (for 3d) or R2LogR (for 2d meshes)

        Examples:
            - [interpolate_field.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolate_field.py)
            - [warp1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp1.py)
            - [warp2.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp2.py)
            - [warp3.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp3.py)
            - [warp4a.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp4a.py)
            - [warp4b.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp4b.py)
            - [warp6.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp6.py)

            ![](https://vedo.embl.es/images/advanced/warp2.png)
        """
        parents = [self]

        try:
            source = source.coordinates
            parents.append(source)
        except AttributeError:
            source = utils.make3d(source)

        try:
            target = target.coordinates
            parents.append(target)
        except AttributeError:
            target = utils.make3d(target)

        ns = len(source)
        nt = len(target)
        if ns != nt:
            vedo.logger.error(f"#source {ns} != {nt} #target points")
            raise RuntimeError()

        NLT = NonLinearTransform(sigma=sigma, mode=mode)
        NLT.source_points = source
        NLT.target_points = target
        self.apply_transform(NLT)

        self.pipeline = utils.OperationNode("warp", parents=parents)
        return self
