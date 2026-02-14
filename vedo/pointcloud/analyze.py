#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""PointAnalyzeMixin extracted from pointcloud core."""

from typing_extensions import Self

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils
from vedo.transformations import LinearTransform, NonLinearTransform
from ._proxy import Points

class PointAnalyzeMixin:
    def compute_normals_with_pca(self, n=20, orientation_point=None, invert=False) -> Self:
        """
        Generate point normals using PCA (principal component analysis).
        This algorithm estimates a local tangent plane around each sample point p
        by considering a small neighborhood of points around p, and fitting a plane
        to the neighborhood (via PCA).

        Arguments:
            n : (int)
                neighborhood size to calculate the normal
            orientation_point : (list)
                adjust the +/- sign of the normals so that
                the normals all point towards a specified point. If None, perform a traversal
                of the point cloud and flip neighboring normals so that they are mutually consistent.
            invert : (bool)
                flip all normals
        """
        poly = self.dataset
        pcan = vtki.new("PCANormalEstimation")
        pcan.SetInputData(poly)
        pcan.SetSampleSize(n)

        if orientation_point is not None:
            pcan.SetNormalOrientationToPoint()
            pcan.SetOrientationPoint(orientation_point)
        else:
            pcan.SetNormalOrientationToGraphTraversal()

        if invert:
            pcan.FlipNormalsOn()
        pcan.Update()

        varr = pcan.GetOutput().GetPointData().GetNormals()
        varr.SetName("Normals")
        self.dataset.GetPointData().SetNormals(varr)
        self.dataset.GetPointData().Modified()
        return self

    def compute_acoplanarity(self, n=25, radius=None, on="points") -> Self:
        """
        Compute acoplanarity which is a measure of how much a local region of the mesh
        differs from a plane.

        The information is stored in a `pointdata` or `celldata` array with name 'Acoplanarity'.

        Either `n` (number of neighbour points) or `radius` (radius of local search) can be specified.
        If a radius value is given and not enough points fall inside it, then a -1 is stored.

        Example:
            ```python
            from vedo import *
            msh = ParametricShape('RandomHills')
            msh.compute_acoplanarity(radius=0.1, on='cells')
            msh.cmap("coolwarm", on='cells').add_scalarbar()
            msh.show(axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/acoplanarity.jpg)
        """
        acoplanarities = []
        if "point" in on:
            pts = self.coordinates
        elif "cell" in on:
            pts = self.cell_centers().coordinates
        else:
            raise ValueError(f"In compute_acoplanarity() set on to either 'cells' or 'points', not {on}")

        for p in utils.progressbar(pts, delay=5, width=15, title=f"{on} acoplanarity"):
            if n:
                data = self.closest_point(p, n=n)
                npts = n
            elif radius:
                data = self.closest_point(p, radius=radius)
                npts = len(data)

            try:
                center = data.mean(axis=0)
                res = np.linalg.svd(data - center)
                acoplanarities.append(res[1][2] / npts)
            except (np.linalg.LinAlgError, ValueError, TypeError, IndexError):
                acoplanarities.append(-1.0)

        if "point" in on:
            self.pointdata["Acoplanarity"] = np.array(acoplanarities, dtype=float)
        else:
            self.celldata["Acoplanarity"] = np.array(acoplanarities, dtype=float)
        return self

    def distance_to(self, pcloud, signed=False, invert=False, name="Distance") -> np.ndarray:
        """
        Computes the distance from one point cloud or mesh to another point cloud or mesh.
        This new `pointdata` array is saved with default name "Distance".

        Keywords `signed` and `invert` are used to compute signed distance,
        but the mesh in that case must have polygonal faces (not a simple point cloud),
        and normals must also be computed.

        Examples:
            - [distance2mesh.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/distance2mesh.py)

                ![](https://vedo.embl.es/images/basic/distance2mesh.png)
        """
        if pcloud.dataset.GetNumberOfPolys():

            poly1 = self.dataset
            poly2 = pcloud.dataset
            df = vtki.new("DistancePolyDataFilter")
            df.ComputeSecondDistanceOff()
            df.SetInputData(0, poly1)
            df.SetInputData(1, poly2)
            df.SetSignedDistance(signed)
            df.SetNegateDistance(invert)
            df.Update()
            scals = df.GetOutput().GetPointData().GetScalars()
            dists = utils.vtk2numpy(scals)

        else:  # has no polygons

            if signed:
                vedo.logger.warning("distance_to() called with signed=True but input object has no polygons")

            if not pcloud.point_locator:
                pcloud.point_locator = vtki.new("PointLocator")
                pcloud.point_locator.SetDataSet(pcloud.dataset)
                pcloud.point_locator.BuildLocator()

            ids = []
            ps1 = self.coordinates
            ps2 = pcloud.coordinates
            for p in ps1:
                pid = pcloud.point_locator.FindClosestPoint(p)
                ids.append(pid)

            deltas = ps2[ids] - ps1
            dists = np.linalg.norm(deltas, axis=1).astype(np.float32)
            scals = utils.numpy2vtk(dists)

        scals.SetName(name)
        self.dataset.GetPointData().AddArray(scals)
        self.dataset.GetPointData().SetActiveScalars(scals.GetName())
        rng = scals.GetRange()
        self.mapper.SetScalarRange(rng[0], rng[1])
        self.mapper.ScalarVisibilityOn()

        self.pipeline = utils.OperationNode(
            "distance_to",
            parents=[self, pcloud],
            shape="cylinder",
            comment=f"#pts {self.dataset.GetNumberOfPoints()}",
        )
        return dists

    def clean(self) -> Self:
        """Clean pointcloud or mesh by removing coincident points."""
        cpd = vtki.new("CleanPolyData")
        cpd.PointMergingOn()
        cpd.ConvertLinesToPointsOff()
        cpd.ConvertPolysToLinesOff()
        cpd.ConvertStripsToPolysOff()
        cpd.SetInputData(self.dataset)
        cpd.Update()
        self._update(cpd.GetOutput())
        self.pipeline = utils.OperationNode(
            "clean", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def subsample(self, fraction: float, absolute=False) -> Self:
        """
        Subsample a point cloud by requiring that the points
        or vertices are far apart at least by the specified fraction of the object size.
        If a Mesh is passed the polygonal faces are not removed
        but holes can appear as their vertices are removed.

        Examples:
            - [moving_least_squares1D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares1D.py)

                ![](https://vedo.embl.es/images/advanced/moving_least_squares1D.png)

            - [recosurface.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/recosurface.py)

                ![](https://vedo.embl.es/images/advanced/recosurface.png)
        """
        if not absolute:
            if fraction > 1:
                vedo.logger.warning(
                    f"subsample(fraction=...), fraction must be < 1, but is {fraction}"
                )
            if fraction <= 0:
                return self

        cpd = vtki.new("CleanPolyData")
        cpd.PointMergingOn()
        cpd.ConvertLinesToPointsOn()
        cpd.ConvertPolysToLinesOn()
        cpd.ConvertStripsToPolysOn()
        cpd.SetInputData(self.dataset)
        if absolute:
            cpd.SetTolerance(fraction / self.diagonal_size())
            # cpd.SetToleranceIsAbsolute(absolute)
        else:
            cpd.SetTolerance(fraction)
        cpd.Update()

        ps = 2
        if self.properties.GetRepresentation() == 0:
            ps = self.properties.GetPointSize()

        self._update(cpd.GetOutput())
        self.ps(ps)

        self.pipeline = utils.OperationNode(
            "subsample", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def threshold(self, scalars: str, above=None, below=None, on="points") -> Self:
        """
        Extracts cells where scalar value satisfies threshold criterion.

        Arguments:
            scalars : (str)
                name of the scalars array.
            above : (float)
                minimum value of the scalar
            below : (float)
                maximum value of the scalar
            on : (str)
                if 'cells' assume array of scalars refers to cell data.

        Examples:
            - [mesh_threshold.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_threshold.py)
        """
        thres = vtki.new("Threshold")
        thres.SetInputData(self.dataset)

        if on.startswith("c"):
            asso = vtki.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            asso = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS

        thres.SetInputArrayToProcess(0, 0, 0, asso, scalars)

        if above is None and below is not None:
            try:  # vtk 9.2
                thres.ThresholdByLower(below)
            except AttributeError:  # vtk 9.3
                thres.SetUpperThreshold(below)

        elif below is None and above is not None:
            try:
                thres.ThresholdByUpper(above)
            except AttributeError:
                thres.SetLowerThreshold(above)
        else:
            try:
                thres.ThresholdBetween(above, below)
            except AttributeError:
                thres.SetUpperThreshold(below)
                thres.SetLowerThreshold(above)

        thres.Update()

        gf = vtki.new("GeometryFilter")
        gf.SetInputData(thres.GetOutput())
        gf.Update()
        self._update(gf.GetOutput())
        self.pipeline = utils.OperationNode("threshold", parents=[self])
        return self

    def quantize(self, value: float) -> Self:
        """
        The user should input a value and all {x,y,z} coordinates
        will be quantized to that absolute grain size.
        """
        qp = vtki.new("QuantizePolyDataPoints")
        qp.SetInputData(self.dataset)
        qp.SetQFactor(value)
        qp.Update()
        self._update(qp.GetOutput())
        self.pipeline = utils.OperationNode("quantize", parents=[self])
        return self

    def vertex_normals(self) -> np.ndarray:
        """
        Retrieve vertex normals as a numpy array. Same as `point_normals`.
        If needed, normals are computed via `compute_normals_with_pca()`.
        Check out also `compute_normals()` and `compute_normals_with_pca()`.
        """
        vtknormals = self.dataset.GetPointData().GetNormals()
        if vtknormals is None:
            self.compute_normals_with_pca()
            vtknormals = self.dataset.GetPointData().GetNormals()
        return utils.vtk2numpy(vtknormals)

    def point_normals(self) -> np.ndarray:
        """
        Retrieve vertex normals as a numpy array. Same as `vertex_normals`.
        Check out also `compute_normals()` and `compute_normals_with_pca()`.
        """
        return self.vertex_normals

    def closest_point(
        self, pt, n=1, radius=None, return_point_id=False, return_cell_id=False
    ) -> list[int] | int | np.ndarray:
        """
        Find the closest point(s) on a mesh given from the input point `pt`.

        Arguments:
            n : (int)
                if greater than 1, return a list of n ordered closest points
            radius : (float)
                if given, get all points within that radius. Then n is ignored.
            return_point_id : (bool)
                return point ID instead of coordinates
            return_cell_id : (bool)
                return cell ID in which the closest point sits

        Examples:
            - [align1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align1.py)
            - [fitplanes.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/fitplanes.py)
            - [quadratic_morphing.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/quadratic_morphing.py)

        .. note::
            The appropriate tree search locator is built on the fly and cached for speed.

            If you want to reset it use `mymesh.point_locator=None`
            and / or `mymesh.cell_locator=None`.
        """
        if len(pt) != 3:
            pt = [pt[0], pt[1], 0]

        # NB: every time the mesh moves or is warped the locators are set to None
        if ((n > 1 or radius) or (n == 1 and return_point_id)) and not return_cell_id:
            poly = None
            if not self.point_locator:
                poly = self.dataset
                self.point_locator = vtki.new("StaticPointLocator")
                self.point_locator.SetDataSet(poly)
                self.point_locator.BuildLocator()

            ##########
            if radius:
                vtklist = vtki.vtkIdList()
                self.point_locator.FindPointsWithinRadius(radius, pt, vtklist)
            elif n > 1:
                vtklist = vtki.vtkIdList()
                self.point_locator.FindClosestNPoints(n, pt, vtklist)
            else:  # n==1 hence return_point_id==True
                ########
                return self.point_locator.FindClosestPoint(pt)
                ########

            if return_point_id:
                ########
                return utils.vtk2numpy(vtklist)
                ########

            if not poly:
                poly = self.dataset
            trgp = []
            for i in range(vtklist.GetNumberOfIds()):
                trgp_ = [0, 0, 0]
                vi = vtklist.GetId(i)
                poly.GetPoints().GetPoint(vi, trgp_)
                trgp.append(trgp_)
            ########
            return np.array(trgp)
            ########

        else:

            if not self.cell_locator:
                poly = self.dataset

                # As per Miquel example with limbs the vtkStaticCellLocator doesnt work !!
                # https://discourse.vtk.org/t/vtkstaticcelllocator-problem-vtk9-0-3/7854/4
                if vedo.vtk_version[0] >= 9 and vedo.vtk_version[1] > 0:
                    self.cell_locator = vtki.new("StaticCellLocator")
                else:
                    self.cell_locator = vtki.new("CellLocator")

                self.cell_locator.SetDataSet(poly)
                self.cell_locator.BuildLocator()

            if radius is not None:
                vedo.printc("Warning: closest_point() with radius is not implemented for cells.", c='r')

            if n != 1:
                vedo.printc("Warning: closest_point() with n>1 is not implemented for cells.", c='r')

            trgp = [0, 0, 0]
            cid = vtki.mutable(0)
            dist2 = vtki.mutable(0)
            subid = vtki.mutable(0)
            self.cell_locator.FindClosestPoint(pt, trgp, cid, subid, dist2)

            if return_cell_id:
                return int(cid)

            return np.array(trgp)

    def auto_distance(self) -> np.ndarray:
        """
        Calculate the distance to the closest point in the same cloud of points.
        The output is stored in a new pointdata array called "AutoDistance",
        and it is also returned by the function.
        """
        points = self.coordinates
        if not self.point_locator:
            self.point_locator = vtki.new("StaticPointLocator")
            self.point_locator.SetDataSet(self.dataset)
            self.point_locator.BuildLocator()
        qs = []
        vtklist = vtki.vtkIdList()
        vtkpoints = self.dataset.GetPoints()
        for p in points:
            self.point_locator.FindClosestNPoints(2, p, vtklist)
            q = [0, 0, 0]
            pid = vtklist.GetId(1)
            vtkpoints.GetPoint(pid, q)
            qs.append(q)
        dists = np.linalg.norm(points - np.array(qs), axis=1)
        self.pointdata["AutoDistance"] = dists
        return dists

    def hausdorff_distance(self, points) -> float:
        """
        Compute the Hausdorff distance to the input point set.
        Returns a single `float`.

        Example:
            ```python
            from vedo import *
            t = np.linspace(0, 2*np.pi, 100)
            x = 4/3 * sin(t)**3
            y = cos(t) - cos(2*t)/3 - cos(3*t)/6 - cos(4*t)/12
            pol1 = Line(np.c_[x,y], closed=True).triangulate()
            pol2 = Polygon(nsides=5).pos(2,2)
            d12 = pol1.distance_to(pol2)
            d21 = pol2.distance_to(pol1)
            pol1.lw(0).cmap("viridis")
            pol2.lw(0).cmap("viridis")
            print("distance d12, d21 :", min(d12), min(d21))
            print("hausdorff distance:", pol1.hausdorff_distance(pol2))
            print("chamfer distance  :", pol1.chamfer_distance(pol2))
            show(pol1, pol2, axes=1)
            ```
            ![](https://vedo.embl.es/images/feats/heart.png)
        """
        hp = vtki.new("HausdorffDistancePointSetFilter")
        hp.SetInputData(0, self.dataset)
        hp.SetInputData(1, points.dataset)
        hp.SetTargetDistanceMethodToPointToCell()
        hp.Update()
        return hp.GetHausdorffDistance()

    def chamfer_distance(self, pcloud) -> float:
        """
        Compute the Chamfer distance to the input point set.

        Example:
            ```python
            from vedo import *
            cloud1 = np.random.randn(1000, 3)
            cloud2 = np.random.randn(1000, 3) + [1, 2, 3]
            c1 = Points(cloud1, r=5, c="red")
            c2 = Points(cloud2, r=5, c="green")
            d = c1.chamfer_distance(c2)
            show(f"Chamfer distance = {d}", c1, c2, axes=1).close()
            ```
        """
        # Definition of Chamfer distance may vary, here we use the average
        if not pcloud.point_locator:
            pcloud.point_locator = vtki.new("PointLocator")
            pcloud.point_locator.SetDataSet(pcloud.dataset)
            pcloud.point_locator.BuildLocator()
        if not self.point_locator:
            self.point_locator = vtki.new("PointLocator")
            self.point_locator.SetDataSet(self.dataset)
            self.point_locator.BuildLocator()

        ps1 = self.coordinates
        ps2 = pcloud.coordinates

        ids12 = []
        for p in ps1:
            pid12 = pcloud.point_locator.FindClosestPoint(p)
            ids12.append(pid12)
        deltav = ps2[ids12] - ps1
        da = np.mean(np.linalg.norm(deltav, axis=1))

        ids21 = []
        for p in ps2:
            pid21 = self.point_locator.FindClosestPoint(p)
            ids21.append(pid21)
        deltav = ps1[ids21] - ps2
        db = np.mean(np.linalg.norm(deltav, axis=1))
        return (da + db) / 2

    def remove_outliers(self, radius: float, neighbors=5) -> Self:
        """
        Remove outliers from a cloud of points within the specified `radius` search.

        Arguments:
            radius : (float)
                Specify the local search radius.
            neighbors : (int)
                Specify the number of neighbors that a point must have,
                within the specified radius, for the point to not be considered isolated.

        Examples:
            - [clustering.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/clustering.py)

                ![](https://vedo.embl.es/images/basic/clustering.png)
        """
        removal = vtki.new("RadiusOutlierRemoval")
        removal.SetInputData(self.dataset)
        removal.SetRadius(radius)
        removal.SetNumberOfNeighbors(neighbors)
        removal.GenerateOutliersOff()
        removal.Update()
        inputobj = removal.GetOutput()
        if inputobj.GetNumberOfCells() == 0:
            carr = vtki.vtkCellArray()
            for i in range(inputobj.GetNumberOfPoints()):
                carr.InsertNextCell(1)
                carr.InsertCellPoint(i)
            inputobj.SetVerts(carr)
        self._update(removal.GetOutput())
        self.pipeline = utils.OperationNode("remove_outliers", parents=[self])
        return self

    def relax_point_positions(
            self,
            n=10,
            iters=10,
            sub_iters=10,
            packing_factor=1,
            max_step=0,
            constraints=(),
        ) -> Self:
        """
        Smooth mesh or points with a
        [Laplacian algorithm](https://vtk.org/doc/nightly/html/classvtkPointSmoothingFilter.html)
        variant. This modifies the coordinates of the input points by adjusting their positions
        to create a smooth distribution (and thereby form a pleasing packing of the points).
        Smoothing is performed by considering the effects of neighboring points on one another
        it uses a cubic cutoff function to produce repulsive forces between close points
        and attractive forces that are a little further away.

        In general, the larger the neighborhood size, the greater the reduction in high frequency
        information. The memory and computational requirements of the algorithm may also
        significantly increase.

        The algorithm incrementally adjusts the point positions through an iterative process.
        Basically points are moved due to the influence of neighboring points.

        As points move, both the local connectivity and data attributes associated with each point
        must be updated. Rather than performing these expensive operations after every iteration,
        a number of sub-iterations can be specified. If so, then the neighborhood and attribute
        value updates occur only every sub iteration, which can improve performance significantly.

        Arguments:
            n : (int)
                neighborhood size to calculate the Laplacian.
            iters : (int)
                number of iterations.
            sub_iters : (int)
                number of sub-iterations, i.e. the number of times the neighborhood and attribute
                value updates occur during each iteration.
            packing_factor : (float)
                adjust convergence speed.
            max_step : (float)
                Specify the maximum smoothing step size for each smoothing iteration.
                This limits the the distance over which a point can move in each iteration.
                As in all iterative methods, the stability of the process is sensitive to this parameter.
                In general, small step size and large numbers of iterations are more stable than a larger
                step size and a smaller numbers of iterations.
            constraints : (dict)
                dictionary of constraints.
                Point constraints are used to prevent points from moving,
                or to move only on a plane. This can prevent shrinking or growing point clouds.
                If enabled, a local topological analysis is performed to determine whether a point
                should be marked as fixed" i.e., never moves, or the point only moves on a plane,
                or the point can move freely.
                If all points in the neighborhood surrounding a point are in the cone defined by
                `fixed_angle`, then the point is classified as fixed.
                If all points in the neighborhood surrounding a point are in the cone defined by
                `boundary_angle`, then the point is classified as lying on a plane.
                Angles are expressed in degrees.

        Example:
            ```py
            import numpy as np
            from vedo import Points, show
            from vedo.pyplot import histogram

            vpts1 = Points(np.random.rand(10_000, 3))
            dists = vpts1.auto_distance()
            h1 = histogram(dists, xlim=(0,0.08)).clone2d()

            vpts2 = vpts1.clone().relax_point_positions(n=100, iters=20, sub_iters=10)
            dists = vpts2.auto_distance()
            h2 = histogram(dists, xlim=(0,0.08)).clone2d()

            show([[vpts1, h1], [vpts2, h2]], N=2).close()
            ```
        """
        smooth = vtki.new("PointSmoothingFilter")
        smooth.SetInputData(self.dataset)
        smooth.SetSmoothingModeToUniform()
        smooth.SetNumberOfIterations(iters)
        smooth.SetNumberOfSubIterations(sub_iters)
        smooth.SetPackingFactor(packing_factor)
        if self.point_locator:
            smooth.SetLocator(self.point_locator)
        if not max_step:
            max_step = self.diagonal_size() / 100
        smooth.SetMaximumStepSize(max_step)
        smooth.SetNeighborhoodSize(n)
        if constraints:
            fixed_angle = constraints.get("fixed_angle", 45)
            boundary_angle = constraints.get("boundary_angle", 110)
            smooth.EnableConstraintsOn()
            smooth.SetFixedAngle(fixed_angle)
            smooth.SetBoundaryAngle(boundary_angle)
            smooth.GenerateConstraintScalarsOn()
            smooth.GenerateConstraintNormalsOn()
        smooth.Update()
        self._update(smooth.GetOutput())
        self.metadata["PackingRadius"] = smooth.GetPackingRadius()
        self.pipeline = utils.OperationNode("relax_point_positions", parents=[self])
        return self

    def smooth_mls_1d(self, f=0.2, radius=None, n=0) -> Self:
        """
        Smooth mesh or points with a `Moving Least Squares` variant.
        The point data array "Variances" will contain the residue calculated for each point.

        Arguments:
            f : (float)
                smoothing factor - typical range is [0,2].
            radius : (float)
                radius search in absolute units.
                If set then `f` is ignored.
            n : (int)
                number of neighbours to be used for the fit.
                If set then `f` and `radius` are ignored.

        Examples:
            - [moving_least_squares1D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares1D.py)
            - [skeletonize.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/skeletonize.py)

            ![](https://vedo.embl.es/images/advanced/moving_least_squares1D.png)
        """
        coords = self.coordinates
        ncoords = len(coords)

        if n:
            Ncp = n
        elif radius:
            Ncp = 1
        else:
            Ncp = int(ncoords * f / 10)
            if Ncp < 5:
                vedo.logger.warning(f"Please choose a fraction higher than {f}")
                Ncp = 5

        variances, newline = [], []
        for p in coords:
            points = self.closest_point(p, n=Ncp, radius=radius)
            if len(points) < 4:
                continue

            points = np.array(points)
            pointsmean = points.mean(axis=0)  # plane center
            _, dd, vv = np.linalg.svd(points - pointsmean)
            newp = np.dot(p - pointsmean, vv[0]) * vv[0] + pointsmean
            variances.append(dd[1] + dd[2])
            newline.append(newp)

        self.pointdata["Variances"] = np.array(variances).astype(np.float32)
        self.coordinates = newline
        self.pipeline = utils.OperationNode("smooth_mls_1d", parents=[self])
        return self

    def smooth_mls_2d(self, f=0.2, radius=None, n=0) -> Self:
        """
        Smooth mesh or points with a `Moving Least Squares` algorithm variant.

        The `mesh.pointdata['MLSVariance']` array will contain the residue calculated for each point.
        When a radius is specified, points that are isolated will not be moved and will get
        a 0 entry in array `mesh.pointdata['MLSValidPoint']`.

        Arguments:
            f : (float)
                smoothing factor - typical range is [0, 2].
            radius : (float | array)
                radius search in absolute units. Can be single value (float) or sequence
                for adaptive smoothing. If set then `f` is ignored.
            n : (int)
                number of neighbours to be used for the fit.
                If set then `f` and `radius` are ignored.

        Examples:
            - [moving_least_squares2D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares2D.py)
            - [recosurface.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/recosurface.py)

                ![](https://vedo.embl.es/images/advanced/recosurface.png)
        """
        coords = self.coordinates
        ncoords = len(coords)

        if n:
            Ncp = n
            radius = None
        elif radius is not None:
            Ncp = 1
        else:
            Ncp = int(ncoords * f / 100)
            if Ncp < 4:
                vedo.logger.error(f"please choose a f-value higher than {f}")
                Ncp = 4

        variances, newpts, valid = [], [], []
        radius_is_sequence = utils.is_sequence(radius)

        pb = None
        if ncoords > 10000:
            pb = utils.ProgressBar(0, ncoords, delay=3)

        for i, p in enumerate(coords):
            if pb:
                pb.print("smooth_mls_2d working ...")

            # if a radius was provided for each point
            if radius_is_sequence:
                pts = self.closest_point(p, n=Ncp, radius=radius[i])
            else:
                pts = self.closest_point(p, n=Ncp, radius=radius)

            if len(pts) > 3:
                ptsmean = pts.mean(axis=0)  # plane center
                _, dd, vv = np.linalg.svd(pts - ptsmean)
                cv = np.cross(vv[0], vv[1])
                t = (np.dot(cv, ptsmean) - np.dot(cv, p)) / np.dot(cv, cv)
                newpts.append(p + cv * t)
                variances.append(dd[2])
                if radius is not None:
                    valid.append(1)
            else:
                newpts.append(p)
                variances.append(0)
                if radius is not None:
                    valid.append(0)

        if radius is not None:
            self.pointdata["MLSValidPoint"] = np.array(valid).astype(np.uint8)
        self.pointdata["MLSVariance"] = np.array(variances).astype(np.float32)

        self.coordinates = newpts

        self.pipeline = utils.OperationNode("smooth_mls_2d", parents=[self])
        return self

    def smooth_lloyd_2d(self, iterations=2, bounds=None, options="Qbb Qc Qx") -> Self:
        """
        Lloyd relaxation of a 2D pointcloud.

        Arguments:
            iterations : (int)
                number of iterations.
            bounds : (list)
                bounding box of the domain.
            options : (str)
                options for the Qhull algorithm.
        """
        # Credits: https://hatarilabs.com/ih-en/
        # tutorial-to-create-a-geospatial-voronoi-sh-mesh-with-python-scipy-and-geopandas
        from scipy.spatial import Voronoi as scipy_voronoi

        def _constrain_points(points):
            # Update any points that have drifted beyond the boundaries of this space
            if bounds is not None:
                for point in points:
                    if point[0] < bounds[0]: point[0] = bounds[0]
                    if point[0] > bounds[1]: point[0] = bounds[1]
                    if point[1] < bounds[2]: point[1] = bounds[2]
                    if point[1] > bounds[3]: point[1] = bounds[3]
            return points

        def _find_centroid(vertices):
            # The equation for the method used here to find the centroid of a
            # 2D polygon is given here: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
            area = 0
            centroid_x = 0
            centroid_y = 0
            for i in range(len(vertices) - 1):
                step = (vertices[i, 0] * vertices[i + 1, 1]) - (vertices[i + 1, 0] * vertices[i, 1])
                centroid_x += (vertices[i, 0] + vertices[i + 1, 0]) * step
                centroid_y += (vertices[i, 1] + vertices[i + 1, 1]) * step
                area += step
            if area:
                centroid_x = (1.0 / (3.0 * area)) * centroid_x
                centroid_y = (1.0 / (3.0 * area)) * centroid_y
            # prevent centroids from escaping bounding box
            return _constrain_points([[centroid_x, centroid_y]])[0]

        def _relax(voron):
            # Moves each point to the centroid of its cell in the voronoi
            # map to "relax" the points (i.e. jitter the points so as
            # to spread them out within the space).
            centroids = []
            for idx in voron.point_region:
                # the region is a series of indices into voronoi.vertices
                # remove point at infinity, designated by index -1
                region = [i for i in voron.regions[idx] if i != -1]
                # enclose the polygon
                region = region + [region[0]]
                verts = voron.vertices[region]
                # find the centroid of those vertices
                centroids.append(_find_centroid(verts))
            return _constrain_points(centroids)

        if bounds is None:
            bounds = self.bounds()

        pts = self.vertices[:, (0, 1)]
        for i in range(iterations):
            vor = scipy_voronoi(pts, qhull_options=options)
            _constrain_points(vor.vertices)
            pts = _relax(vor)
        out = Points(pts)
        out.name = "MeshSmoothLloyd2D"
        out.pipeline = utils.OperationNode("smooth_lloyd", parents=[self])
        return out

    def compute_clustering(self, radius: float) -> Self:
        """
        Cluster points in space. The `radius` is the radius of local search.

        An array named "ClusterId" is added to `pointdata`.

        Examples:
            - [clustering.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/clustering.py)

                ![](https://vedo.embl.es/images/basic/clustering.png)
        """
        cluster = vtki.new("EuclideanClusterExtraction")
        cluster.SetInputData(self.dataset)
        cluster.SetExtractionModeToAllClusters()
        cluster.SetRadius(radius)
        cluster.ColorClustersOn()
        cluster.Update()
        idsarr = cluster.GetOutput().GetPointData().GetArray("ClusterId")
        self.dataset.GetPointData().AddArray(idsarr)
        self.pipeline = utils.OperationNode(
            "compute_clustering", parents=[self], comment=f"radius = {radius}"
        )
        return self

    def compute_connections(self, radius, mode=0, regions=(), vrange=(0, 1), seeds=(), angle=0.0) -> Self:
        """
        Extracts and/or segments points from a point cloud based on geometric distance measures
        (e.g., proximity, normal alignments, etc.) and optional measures such as scalar range.
        The default operation is to segment the points into "connected" regions where the connection
        is determined by an appropriate distance measure. Each region is given a region id.

        Optionally, the filter can output the largest connected region of points; a particular region
        (via id specification); those regions that are seeded using a list of input point ids;
        or the region of points closest to a specified position.

        The key parameter of this filter is the radius defining a sphere around each point which defines
        a local neighborhood: any other points in the local neighborhood are assumed connected to the point.
        Note that the radius is defined in absolute terms.

        Other parameters are used to further qualify what it means to be a neighboring point.
        For example, scalar range and/or point normals can be used to further constrain the neighborhood.
        Also the extraction mode defines how the filter operates.
        By default, all regions are extracted but it is possible to extract particular regions;
        the region closest to a seed point; seeded regions; or the largest region found while processing.
        By default, all regions are extracted.

        On output, all points are labeled with a region number.
        However note that the number of input and output points may not be the same:
        if not extracting all regions then the output size may be less than the input size.

        Arguments:
            radius : (float)
                variable specifying a local sphere used to define local point neighborhood
            mode : (int)
                - 0,  Extract all regions
                - 1,  Extract point seeded regions
                - 2,  Extract largest region
                - 3,  Test specified regions
                - 4,  Extract all regions with scalar connectivity
                - 5,  Extract point seeded regions
            regions : (list)
                a list of non-negative regions id to extract
            vrange : (list)
                scalar range to use to extract points based on scalar connectivity
            seeds : (list)
                a list of non-negative point seed ids
            angle : (list)
                points are connected if the angle between their normals is
                within this angle threshold (expressed in degrees).
        """
        # https://vtk.org/doc/nightly/html/classvtkConnectedPointsFilter.html
        cpf = vtki.new("ConnectedPointsFilter")
        cpf.SetInputData(self.dataset)
        cpf.SetRadius(radius)
        if mode == 0:  # Extract all regions
            pass

        elif mode == 1:  # Extract point seeded regions
            cpf.SetExtractionModeToPointSeededRegions()
            for s in seeds:
                cpf.AddSeed(s)

        elif mode == 2:  # Test largest region
            cpf.SetExtractionModeToLargestRegion()

        elif mode == 3:  # Test specified regions
            cpf.SetExtractionModeToSpecifiedRegions()
            for r in regions:
                cpf.AddSpecifiedRegion(r)

        elif mode == 4:  # Extract all regions with scalar connectivity
            cpf.SetExtractionModeToLargestRegion()
            cpf.ScalarConnectivityOn()
            cpf.SetScalarRange(vrange[0], vrange[1])

        elif mode == 5:  # Extract point seeded regions
            cpf.SetExtractionModeToLargestRegion()
            cpf.ScalarConnectivityOn()
            cpf.SetScalarRange(vrange[0], vrange[1])
            cpf.AlignedNormalsOn()
            cpf.SetNormalAngle(angle)

        cpf.Update()
        self._update(cpf.GetOutput(), reset_locators=False)
        return self

    def compute_camera_distance(self) -> np.ndarray:
        """
        Calculate the distance from points to the camera.

        A pointdata array is created with name 'DistanceToCamera' and returned.
        """
        plt = vedo.current_plotter()
        if plt and plt.renderer:
            poly = self.dataset
            dc = vtki.new("DistanceToCamera")
            dc.SetInputData(poly)
            dc.SetRenderer(plt.renderer)
            dc.Update()
            self._update(dc.GetOutput(), reset_locators=False)
            return self.pointdata["DistanceToCamera"]
        return np.array([])

    def densify(self, target_distance=0.1, nclosest=6, radius=None, niter=1, nmax=None) -> Self:
        """
        Return a copy of the cloud with new added points.
        The new points are created in such a way that all points in any local neighborhood are
        within a target distance of one another.

        For each input point, the distance to all points in its neighborhood is computed.
        If any of its neighbors is further than the target distance,
        the edge connecting the point and its neighbor is bisected and
        a new point is inserted at the bisection point.
        A single pass is completed once all the input points are visited.
        Then the process repeats to the number of iterations.

        Examples:
            - [densifycloud.py](https://github.com/marcomusy/vedo/blob/master/examples/volumetric/densifycloud.py)

                ![](https://vedo.embl.es/images/volumetric/densifycloud.png)

        .. note::
            Points will be created in an iterative fashion until all points in their
            local neighborhood are the target distance apart or less.
            Note that the process may terminate early due to the
            number of iterations. By default the target distance is set to 0.5.
            Note that the target_distance should be less than the radius
            or nothing will change on output.

        .. warning::
            This class can generate a lot of points very quickly.
            The maximum number of iterations is by default set to =1.0 for this reason.
            Increase the number of iterations very carefully.
            Also, `nmax` can be set to limit the explosion of points.
            It is also recommended that a N closest neighborhood is used.

        """
        src = vtki.new("ProgrammableSource")
        opts = self.coordinates
        # zeros = np.zeros(3)

        def _read_points():
            output = src.GetPolyDataOutput()
            points = vtki.vtkPoints()
            for p in opts:
                # print(p)
                # if not np.array_equal(p, zeros):
                points.InsertNextPoint(p)
            output.SetPoints(points)

        src.SetExecuteMethod(_read_points)

        dens = vtki.new("DensifyPointCloudFilter")
        dens.SetInputConnection(src.GetOutputPort())
        # dens.SetInputData(self.dataset) # this does not work
        dens.InterpolateAttributeDataOn()
        dens.SetTargetDistance(target_distance)
        dens.SetMaximumNumberOfIterations(niter)
        if nmax:
            dens.SetMaximumNumberOfPoints(nmax)

        if radius:
            dens.SetNeighborhoodTypeToRadius()
            dens.SetRadius(radius)
        elif nclosest:
            dens.SetNeighborhoodTypeToNClosest()
            dens.SetNumberOfClosestPoints(nclosest)
        else:
            vedo.logger.error("set either radius or nclosest")
            raise RuntimeError()
        dens.Update()

        cld = Points(dens.GetOutput())
        cld.copy_properties_from(self)
        cld.interpolate_data_from(self, n=nclosest, radius=radius)
        cld.name = "DensifiedCloud"
        cld.pipeline = utils.OperationNode(
            "densify",
            parents=[self],
            c="#e9c46a:",
            comment=f"#pts {cld.dataset.GetNumberOfPoints()}",
        )
        return cld

    def density(
        self, dims=(40, 40, 40), bounds=None, radius=None, compute_gradient=False, locator=None
    ) -> "vedo.Volume":
        """
        Generate a density field from a point cloud. Input can also be a set of 3D coordinates.
        Output is a `Volume`.

        The local neighborhood is specified as the `radius` around each sample position (each voxel).
        If left to None, the radius is automatically computed as the diagonal of the bounding box
        and can be accessed via `vol.metadata["radius"]`.
        The density is expressed as the number of counts in the radius search.

        Arguments:
            dims : (int, list)
                number of voxels in x, y and z of the output Volume.
            compute_gradient : (bool)
                Turn on/off the generation of the gradient vector,
                gradient magnitude scalar, and function classification scalar.
                By default this is off. Note that this will increase execution time
                and the size of the output. (The names of these point data arrays are:
                "Gradient", "Gradient Magnitude", and "Classification")
            locator : (vtkPointLocator)
                can be assigned from a previous call for speed (access it via `object.point_locator`).

        Examples:
            - [plot_density3d.py](https://github.com/marcomusy/vedo/blob/master/examples/pyplot/plot_density3d.py)

                ![](https://vedo.embl.es/images/pyplot/plot_density3d.png)
        """
        pdf = vtki.new("PointDensityFilter")
        pdf.SetInputData(self.dataset)

        if not utils.is_sequence(dims):
            dims = [dims, dims, dims]

        if bounds is None:
            bounds = list(self.bounds())
        elif len(bounds) == 4:
            bounds = [*bounds, 0, 0]

        if bounds[5] - bounds[4] == 0 or len(dims) == 2:  # its 2D
            dims = list(dims)
            dims = [dims[0], dims[1], 2]
            diag = self.diagonal_size()
            bounds[5] = bounds[4] + diag / 1000
        pdf.SetModelBounds(bounds)

        pdf.SetSampleDimensions(dims)

        if locator:
            pdf.SetLocator(locator)

        pdf.SetDensityEstimateToFixedRadius()
        if radius is None:
            radius = self.diagonal_size() / 20
        pdf.SetRadius(radius)
        pdf.SetComputeGradient(compute_gradient)
        pdf.Update()

        vol = vedo.Volume(pdf.GetOutput()).mode(1)
        vol.name = "PointDensity"
        vol.metadata["radius"] = radius
        vol.locator = pdf.GetLocator()
        vol.pipeline = utils.OperationNode(
            "density", parents=[self], comment=f"dims={tuple(vol.dimensions())}"
        )
        return vol

    def visible_points(self, area=(), tol=None, invert=False) -> Self | None:
        """
        Extract points based on whether they are visible or not.
        Visibility is determined by accessing the z-buffer of a rendering window.
        The position of each input point is converted into display coordinates,
        and then the z-value at that point is obtained.
        If within the user-specified tolerance, the point is considered visible.
        Associated data attributes are passed to the output as well.

        This filter also allows you to specify a rectangular window in display (pixel)
        coordinates in which the visible points must lie.

        Arguments:
            area : (list)
                specify a rectangular region as (xmin,xmax,ymin,ymax)
            tol : (float)
                a tolerance in normalized display coordinate system
            invert : (bool)
                select invisible points instead.

        Example:
            ```python
            from vedo import Ellipsoid, show
            s = Ellipsoid().rotate_y(30)

            # Camera options: pos, focal_point, viewup, distance
            camopts = dict(pos=(0,0,25), focal_point=(0,0,0))
            show(s, camera=camopts, offscreen=True)

            m = s.visible_points()
            # print('visible pts:', m.vertices)  # numpy array
            show(m, new=True, axes=1).close() # optionally draw result in a new window
            ```
            ![](https://vedo.embl.es/images/feats/visible_points.png)
        """
        svp = vtki.new("SelectVisiblePoints")
        svp.SetInputData(self.dataset)

        ren = None
        plt = vedo.current_plotter()
        if plt and plt.renderer:
            ren = plt.renderer
            svp.SetRenderer(ren)
        if not ren:
            vedo.logger.warning(
                "visible_points() can only be used after a rendering step"
            )
            return None

        if len(area) == 2:
            area = utils.flatten(area)
        if len(area) == 4:
            # specify a rectangular region
            svp.SetSelection(area[0], area[1], area[2], area[3])
        if tol is not None:
            svp.SetTolerance(tol)
        if invert:
            svp.SelectInvisibleOn()
        svp.Update()

        m = Points(svp.GetOutput())
        m.name = "VisiblePoints"
        return m

