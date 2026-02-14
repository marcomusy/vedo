#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Quality/analysis mixin for Mesh."""

from typing_extensions import Self

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo.utils import vtk2numpy, OperationNode
from ._proxy import Mesh


class MeshMetricsMixin:
    def vertex_normals(self) -> np.ndarray:
        """
        Retrieve vertex normals as a numpy array. 
        If needed, normals are automatically computed via `compute_normals()`.
        Check out also `compute_normals_with_pca()`.
        """
        vtknormals = self.dataset.GetPointData().GetNormals()
        if vtknormals is None:
            self.compute_normals()
            vtknormals = self.dataset.GetPointData().GetNormals()
        return vtk2numpy(vtknormals)


    def cell_normals(self) -> np.ndarray:
        """
        Retrieve face normals as a numpy array.
        If need be normals are computed via `compute_normals()`.
        Check out also `compute_normals(cells=True)` and `compute_normals_with_pca()`.
        """
        vtknormals = self.dataset.GetCellData().GetNormals()
        if vtknormals is None:
            self.compute_normals()
            vtknormals = self.dataset.GetCellData().GetNormals()
        return vtk2numpy(vtknormals)


    def compute_normals(self, points=True, cells=True, feature_angle=None, consistency=True) -> Self:
        """
        Compute cell and vertex normals for the mesh.

        Arguments:
            points : (bool)
                do the computation for the vertices too
            cells : (bool)
                do the computation for the cells too
            feature_angle : (float)
                specify the angle that defines a sharp edge.
                If the difference in angle across neighboring polygons is greater than this value,
                the shared edge is considered "sharp" and it is split.
            consistency : (bool)
                turn on/off the enforcement of consistent polygon ordering.

        .. warning::
            If `feature_angle` is set then the Mesh can be modified, and it
            can have a different number of vertices from the original.

            Note that the appearance of the mesh may change if the normals are computed,
            as shading is automatically enabled when such information is present.
            Use `mesh.flat()` to avoid smoothing effects.
        """
        pdnorm = vtki.new("PolyDataNormals")
        pdnorm.SetInputData(self.dataset)
        pdnorm.SetComputePointNormals(points)
        pdnorm.SetComputeCellNormals(cells)
        pdnorm.SetConsistency(consistency)
        pdnorm.FlipNormalsOff()
        if feature_angle:
            pdnorm.SetSplitting(True)
            pdnorm.SetFeatureAngle(feature_angle)
        else:
            pdnorm.SetSplitting(False)
        pdnorm.Update()
        out = pdnorm.GetOutput()
        self._update(out, reset_locators=False)
        return self


    def volume(self) -> float:
        """
        Compute the volume occupied by mesh.
        The mesh must be triangular for this to work.
        To triangulate a mesh use `mesh.triangulate()`.
        """
        mass = vtki.new("MassProperties")
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.dataset)
        mass.Update()
        mass.SetGlobalWarningDisplay(1)
        return mass.GetVolume()


    def area(self) -> float:
        """
        Compute the surface area of the mesh.
        The mesh must be triangular for this to work.
        To triangulate a mesh use `mesh.triangulate()`.
        """
        mass = vtki.new("MassProperties")
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.dataset)
        mass.Update()
        mass.SetGlobalWarningDisplay(1)
        return mass.GetSurfaceArea()


    def is_closed(self) -> bool:
        """
        Return `True` if the mesh is watertight.
        Note that if the mesh contains coincident points the result may be flase.
        Use in this case `mesh.clean()` to merge coincident points.
        """
        fe = vtki.new("FeatureEdges")
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOn()
        fe.SetInputData(self.dataset)
        fe.Update()
        ne = fe.GetOutput().GetNumberOfCells()
        return not bool(ne)


    def is_manifold(self) -> bool:
        """Return `True` if the mesh is manifold."""
        fe = vtki.new("FeatureEdges")
        fe.BoundaryEdgesOff()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOn()
        fe.SetInputData(self.dataset)
        fe.Update()
        ne = fe.GetOutput().GetNumberOfCells()
        return not bool(ne)


    def non_manifold_faces(self, remove=True, tol="auto") -> Self:
        """
        Detect and (try to) remove non-manifold faces of a triangular mesh:

            - set `remove` to `False` to mark cells without removing them.
            - set `tol=0` for zero-tolerance, the result will be manifold but with holes.
            - set `tol>0` to cut off non-manifold faces, and try to recover the good ones.
            - set `tol="auto"` to make an automatic choice of the tolerance.
        """
        # mark original point and cell ids
        self.add_ids()
        toremove = self.boundaries(
            boundary_edges=False,
            non_manifold_edges=True,
            cell_edge=True,
            return_cell_ids=True,
        )
        if len(toremove) == 0: # type: ignore
            return self

        points = self.coordinates
        faces = self.cells
        centers = self.cell_centers().coordinates

        copy = self.clone()
        copy.delete_cells(toremove).clean()
        copy.compute_normals(cells=False)
        normals = copy.vertex_normals
        deltas, deltas_i = [], []

        for i in vedo.utils.progressbar(toremove, delay=3, title="recover faces"):
            pids = copy.closest_point(centers[i], n=3, return_point_id=True)
            norms = normals[pids]
            n = np.mean(norms, axis=0)
            dn = np.linalg.norm(n)
            if not dn:
                continue
            n = n / dn

            p0, p1, p2 = points[faces[i]][:3]
            v = np.cross(p1 - p0, p2 - p0)
            lv = np.linalg.norm(v)
            if not lv:
                continue
            v = v / lv

            cosa = 1 - np.dot(n, v)
            deltas.append(cosa)
            deltas_i.append(i)

        recover = []
        if len(deltas) > 0:
            mean_delta = np.mean(deltas)
            err_delta = np.std(deltas)
            txt = ""
            if tol == "auto":  # automatic choice
                tol = mean_delta / 5
                txt = f"\n Automatic tol. : {tol: .4f}"
            for i, cosa in zip(deltas_i, deltas):
                if cosa < tol:
                    recover.append(i)

            vedo.logger.info(
                f"\n --------- Non manifold faces ---------"
                f"\n Average tol.   : {mean_delta: .4f} +- {err_delta: .4f}{txt}"
                f"\n Removed faces  : {len(toremove)}" # type: ignore
                f"\n Recovered faces: {len(recover)}"
            )

        toremove = list(set(toremove) - set(recover)) # type: ignore

        if not remove:
            mark = np.zeros(self.ncells, dtype=np.uint8)
            mark[recover] = 1
            mark[toremove] = 2
            self.celldata["NonManifoldCell"] = mark
        else:
            self.delete_cells(toremove) # type: ignore

        self.pipeline = OperationNode(
            "non_manifold_faces",
            parents=[self],
            comment=f"#cells {self.dataset.GetNumberOfCells()}",
        )
        return self


    def euler_characteristic(self) -> int:
        """
        Compute the Euler characteristic of the mesh.
        The Euler characteristic is a topological invariant for surfaces.
        """
        return self.npoints - len(self.edges) + self.ncells


    def genus(self) -> int:
        """
        Compute the genus of the mesh.
        The genus is a topological invariant for surfaces.
        """
        nb = len(self.boundaries().split()) - 1
        return (2 - self.euler_characteristic() - nb ) / 2


    def compute_cell_vertex_count(self) -> Self:
        """
        Add to this mesh a cell data array containing the nr of vertices that a polygonal face has.
        """
        csf = vtki.new("CellSizeFilter")
        csf.SetInputData(self.dataset)
        csf.SetComputeArea(False)
        csf.SetComputeVolume(False)
        csf.SetComputeLength(False)
        csf.SetComputeVertexCount(True)
        csf.SetVertexCountArrayName("VertexCount")
        csf.Update()
        self.dataset.GetCellData().AddArray(
            csf.GetOutput().GetCellData().GetArray("VertexCount")
        )
        return self


    def compute_quality(self, metric=6) -> Self:
        """
        Calculate metrics of quality for the elements of a triangular mesh.
        This method adds to the mesh a cell array named "Quality".
        See class
        [vtkMeshQuality](https://vtk.org/doc/nightly/html/classvtkMeshQuality.html).

        Arguments:
            metric : (int)
                type of available estimators are:
                - EDGE RATIO, 0
                - ASPECT RATIO, 1
                - RADIUS RATIO, 2
                - ASPECT FROBENIUS, 3
                - MED ASPECT FROBENIUS, 4
                - MAX ASPECT FROBENIUS, 5
                - MIN_ANGLE, 6
                - COLLAPSE RATIO, 7
                - MAX ANGLE, 8
                - CONDITION, 9
                - SCALED JACOBIAN, 10
                - SHEAR, 11
                - RELATIVE SIZE SQUARED, 12
                - SHAPE, 13
                - SHAPE AND SIZE, 14
                - DISTORTION, 15
                - MAX EDGE RATIO, 16
                - SKEW, 17
                - TAPER, 18
                - VOLUME, 19
                - STRETCH, 20
                - DIAGONAL, 21
                - DIMENSION, 22
                - ODDY, 23
                - SHEAR AND SIZE, 24
                - JACOBIAN, 25
                - WARPAGE, 26
                - ASPECT GAMMA, 27
                - AREA, 28
                - ASPECT BETA, 29

        Examples:
            - [meshquality.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/meshquality.py)

            ![](https://vedo.embl.es/images/advanced/meshquality.png)
        """
        qf = vtki.new("MeshQuality")
        qf.SetInputData(self.dataset)
        qf.SetTriangleQualityMeasure(metric)
        qf.SaveCellQualityOn()
        qf.Update()
        self._update(qf.GetOutput(), reset_locators=False)
        self.mapper.SetScalarModeToUseCellData()
        self.pipeline = OperationNode("compute_quality", parents=[self])
        return self


    def count_vertices(self) -> np.ndarray:
        """Count the number of vertices each cell has and return it as a numpy array"""
        vc = vtki.new("CountVertices")
        vc.SetInputData(self.dataset)
        vc.SetOutputArrayName("VertexCount")
        vc.Update()
        varr = vc.GetOutput().GetCellData().GetArray("VertexCount")
        return vtk2numpy(varr)


    def check_validity(self, tol=0) -> np.ndarray:
        """
        Return a numpy array of possible problematic faces following this convention:
        - Valid               =  0
        - WrongNumberOfPoints =  1
        - IntersectingEdges   =  2
        - IntersectingFaces   =  4
        - NoncontiguousEdges  =  8
        - Nonconvex           = 10
        - OrientedIncorrectly = 20

        Arguments:
            tol : (float)
                value is used as an epsilon for floating point
                equality checks throughout the cell checking process.
        """
        vald = vtki.new("CellValidator")
        if tol:
            vald.SetTolerance(tol)
        vald.SetInputData(self.dataset)
        vald.Update()
        varr = vald.GetOutput().GetCellData().GetArray("ValidityState")
        return vtk2numpy(varr)


    def compute_curvature(self, method=0) -> Self:
        """
        Add scalars to `Mesh` that contains the curvature calculated in three different ways.

        Variable `method` can be:
        - 0 = gaussian
        - 1 = mean curvature
        - 2 = max curvature
        - 3 = min curvature

        Example:
            ```python
            from vedo import Torus
            Torus().compute_curvature().add_scalarbar().show().close()
            ```
            ![](https://vedo.embl.es/images/advanced/torus_curv.png)
        """
        curve = vtki.new("Curvatures")
        curve.SetInputData(self.dataset)
        curve.SetCurvatureType(method)
        curve.Update()
        self._update(curve.GetOutput(), reset_locators=False)
        self.mapper.ScalarVisibilityOn()
        return self


    def compute_elevation(self, low=(0, 0, 0), high=(0, 0, 1), vrange=(0, 1)) -> Self:
        """
        Add to `Mesh` a scalar array that contains distance along a specified direction.

        Arguments:
            low : (list)
                one end of the line (small scalar values)
            high : (list)
                other end of the line (large scalar values)
            vrange : (list)
                set the range of the scalar

        Example:
            ```python
            from vedo import Sphere
            s = Sphere().compute_elevation(low=(0,0,0), high=(1,1,1))
            s.add_scalarbar().show(axes=1).close()
            ```
            ![](https://vedo.embl.es/images/basic/compute_elevation.png)
        """
        ef = vtki.new("ElevationFilter")
        ef.SetInputData(self.dataset)
        ef.SetLowPoint(low)
        ef.SetHighPoint(high)
        ef.SetScalarRange(vrange)
        ef.Update()
        self._update(ef.GetOutput(), reset_locators=False)
        self.mapper.ScalarVisibilityOn()
        return self


