#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""PointReconstructMixin extracted from pointcloud core."""

from typing_extensions import Self

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils
from vedo.transformations import LinearTransform, NonLinearTransform

class PointReconstructMixin:
    def generate_surface_halo(
            self,
            distance=0.05,
            res=(50, 50, 50),
            bounds=(),
            maxdist=None,
    ) -> vedo.Mesh:
        """
        Generate the surface halo which sits at the specified distance from the input one.

        Arguments:
            distance : (float)
                distance from the input surface
            res : (int)
                resolution of the surface
            bounds : (list)
                bounding box of the surface
            maxdist : (float)
                maximum distance to generate the surface
        """
        if not bounds:
            bounds = self.bounds()

        if not maxdist:
            maxdist = self.diagonal_size() / 2

        imp = vtki.new("ImplicitModeller")
        imp.SetInputData(self.dataset)
        imp.SetSampleDimensions(res)
        if maxdist:
            imp.SetMaximumDistance(maxdist)
        if len(bounds) == 6:
            imp.SetModelBounds(bounds)
        contour = vtki.new("ContourFilter")
        contour.SetInputConnection(imp.GetOutputPort())
        contour.SetValue(0, distance)
        contour.Update()
        out = vedo.Mesh(contour.GetOutput())
        out.c("lightblue").alpha(0.25).lighting("off")
        out.pipeline = utils.OperationNode("generate_surface_halo", parents=[self])
        return out

    def generate_mesh(
        self,
        line_resolution=None,
        mesh_resolution=None,
        smooth=0.0,
        jitter=0.001,
        grid=None,
        quads=False,
        invert=False,
    ) -> Self:
        """
        Generate a polygonal Mesh from a closed contour line.
        If line is not closed it will be closed with a straight segment.

        Check also `generate_delaunay2d()`.

        Arguments:
            line_resolution : (int)
                resolution of the contour line. The default is None, in this case
                the contour is not resampled.
            mesh_resolution : (int)
                resolution of the internal triangles not touching the boundary.
            smooth : (float)
                smoothing of the contour before meshing.
            jitter : (float)
                add a small noise to the internal points.
            grid : (Grid)
                manually pass a Grid object. The default is True.
            quads : (bool)
                generate a mesh of quads instead of triangles.
            invert : (bool)
                flip the line orientation. The default is False.

        Examples:
            - [line2mesh_tri.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/line2mesh_tri.py)

                ![](https://vedo.embl.es/images/advanced/line2mesh_tri.jpg)

            - [line2mesh_quads.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/line2mesh_quads.py)

                ![](https://vedo.embl.es/images/advanced/line2mesh_quads.png)
        """
        if line_resolution is None:
            contour = vedo.shapes.Line(self.coordinates)
        else:
            contour = vedo.shapes.Spline(self.coordinates, smooth=smooth, res=line_resolution)
        contour.clean()

        length = contour.length()
        density = length / contour.npoints
        # print(f"tomesh():\n\tline length = {length}")
        # print(f"\tdensity = {density} length/pt_separation")

        x0, x1 = contour.xbounds()
        y0, y1 = contour.ybounds()

        if grid is None:
            if mesh_resolution is None:
                resx = int((x1 - x0) / density + 0.5)
                resy = int((y1 - y0) / density + 0.5)
                # print(f"tmesh_resolution = {[resx, resy]}")
            else:
                if utils.is_sequence(mesh_resolution):
                    resx, resy = mesh_resolution
                else:
                    resx, resy = mesh_resolution, mesh_resolution
            grid = vedo.shapes.Grid(
                [(x0 + x1) / 2, (y0 + y1) / 2, 0],
                s=((x1 - x0) * 1.025, (y1 - y0) * 1.025),
                res=(resx, resy),
            )
        else:
            grid = grid.clone()

        cpts = contour.coordinates

        # make sure it's closed
        p0, p1 = cpts[0], cpts[-1]
        nj = max(2, int(utils.mag(p1 - p0) / density + 0.5))
        joinline = vedo.shapes.Line(p1, p0, res=nj)
        contour = vedo.merge(contour, joinline).subsample(0.0001)

        ####################################### quads
        if quads:
            cmesh = grid.clone().cut_with_point_loop(contour, on="cells", invert=invert)
            cmesh.wireframe(False).lw(0.5)
            cmesh.pipeline = utils.OperationNode(
                "generate_mesh",
                parents=[self, contour],
                comment=f"#quads {cmesh.dataset.GetNumberOfCells()}",
            )
            return cmesh
        #############################################

        grid_tmp = grid.coordinates.copy()

        if jitter:
            np.random.seed(0)
            sigma = 1.0 / np.sqrt(grid.npoints) * grid.diagonal_size() * jitter
            # print(f"\tsigma jittering = {sigma}")
            grid_tmp += np.random.rand(grid.npoints, 3) * sigma
            grid_tmp[:, 2] = 0.0

        todel = []
        density /= np.sqrt(3)
        vgrid_tmp = vedo.Points(grid_tmp)

        for p in contour.coordinates:
            out = vgrid_tmp.closest_point(p, radius=density, return_point_id=True)
            todel += out.tolist()

        grid_tmp = grid_tmp.tolist()
        for index in sorted(list(set(todel)), reverse=True):
            del grid_tmp[index]

        points = contour.coordinates.tolist() + grid_tmp
        if invert:
            boundary = list(reversed(range(contour.npoints)))
        else:
            boundary = list(range(contour.npoints))

        dln = vedo.Points(points).generate_delaunay2d(mode="xy", boundaries=[boundary])
        dln.compute_normals(points=False)  # fixes reversd faces
        dln.lw(1)

        dln.pipeline = utils.OperationNode(
            "generate_mesh",
            parents=[self, contour],
            comment=f"#cells {dln.dataset.GetNumberOfCells()}",
        )
        return dln

    def reconstruct_surface(
        self,
        dims=(100, 100, 100),
        radius=None,
        sample_size=None,
        hole_filling=True,
        bounds=(),
        padding=0.05,
    ) -> vedo.Mesh:
        """
        Surface reconstruction from a scattered cloud of points.

        Arguments:
            dims : (int)
                number of voxels in x, y and z to control precision.
            radius : (float)
                radius of influence of each point.
                Smaller values generally improve performance markedly.
                Note that after the signed distance function is computed,
                any voxel taking on the value >= radius
                is presumed to be "unseen" or uninitialized.
            sample_size : (int)
                if normals are not present
                they will be calculated using this sample size per point.
            hole_filling : (bool)
                enables hole filling, this generates
                separating surfaces between the empty and unseen portions of the volume.
            bounds : (list)
                region in space in which to perform the sampling
                in format (xmin,xmax, ymin,ymax, zim, zmax)
            padding : (float)
                increase by this fraction the bounding box

        Examples:
            - [recosurface.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/recosurface.py)

                ![](https://vedo.embl.es/images/advanced/recosurface.png)
        """
        if not utils.is_sequence(dims):
            dims = (dims, dims, dims)

        sdf = vtki.new("SignedDistance")

        if len(bounds) == 6:
            sdf.SetBounds(bounds)
        else:
            x0, x1, y0, y1, z0, z1 = self.bounds()
            sdf.SetBounds(
                x0 - (x1 - x0) * padding,
                x1 + (x1 - x0) * padding,
                y0 - (y1 - y0) * padding,
                y1 + (y1 - y0) * padding,
                z0 - (z1 - z0) * padding,
                z1 + (z1 - z0) * padding,
            )

        bb = sdf.GetBounds()
        if bb[0]==bb[1]:
            vedo.logger.warning("reconstruct_surface(): zero x-range")
        if bb[2]==bb[3]:
            vedo.logger.warning("reconstruct_surface(): zero y-range")
        if bb[4]==bb[5]:
            vedo.logger.warning("reconstruct_surface(): zero z-range")

        pd = self.dataset

        if pd.GetPointData().GetNormals():
            sdf.SetInputData(pd)
        else:
            normals = vtki.new("PCANormalEstimation")
            normals.SetInputData(pd)
            if not sample_size:
                sample_size = int(pd.GetNumberOfPoints() / 50)
            normals.SetSampleSize(sample_size)
            normals.SetNormalOrientationToGraphTraversal()
            sdf.SetInputConnection(normals.GetOutputPort())
            # print("Recalculating normals with sample size =", sample_size)

        if radius is None:
            radius = self.diagonal_size() / (sum(dims) / 3) * 5
            # print("Calculating mesh from points with radius =", radius)

        sdf.SetRadius(radius)
        sdf.SetDimensions(dims)
        sdf.Update()

        surface = vtki.new("ExtractSurface")
        surface.SetRadius(radius * 0.99)
        surface.SetHoleFilling(hole_filling)
        surface.ComputeNormalsOff()
        surface.ComputeGradientsOff()
        surface.SetInputConnection(sdf.GetOutputPort())
        surface.Update()
        m = vedo.mesh.Mesh(surface.GetOutput(), c=self.color())

        m.pipeline = utils.OperationNode(
            "reconstruct_surface",
            parents=[self],
            comment=f"#pts {m.dataset.GetNumberOfPoints()}",
        )
        return m

    def tovolume(
        self,
        kernel="shepard",
        radius=None,
        n=None,
        bounds=None,
        null_value=None,
        dims=(25, 25, 25),
    ) -> vedo.Volume:
        """
        Generate a `Volume` by interpolating a scalar
        or vector field which is only known on a scattered set of points or mesh.
        Available interpolation kernels are: shepard, gaussian, or linear.

        Arguments:
            kernel : (str)
                interpolation kernel type [shepard]
            radius : (float)
                radius of the local search
            n : (int)
                number of point to use for interpolation
            bounds : (list)
                bounding box of the output Volume object
            dims : (list)
                dimensions of the output Volume object
            null_value : (float)
                value to be assigned to invalid points

        Examples:
            - [interpolate_volume.py](https://github.com/marcomusy/vedo/blob/master/examples/volumetric/interpolate_volume.py)

                ![](https://vedo.embl.es/images/volumetric/59095175-1ec5a300-8918-11e9-8bc0-fd35c8981e2b.jpg)
        """
        if radius is None and not n:
            vedo.logger.error("please set either radius or n")
            raise RuntimeError

        poly = self.dataset

        # Create a probe volume
        probe = vtki.vtkImageData()
        probe.SetDimensions(dims)
        if bounds is None:
            bounds = self.bounds()
        probe.SetOrigin(bounds[0], bounds[2], bounds[4])
        probe.SetSpacing(
            (bounds[1] - bounds[0]) / dims[0],
            (bounds[3] - bounds[2]) / dims[1],
            (bounds[5] - bounds[4]) / dims[2],
        )

        if not self.point_locator:
            self.point_locator = vtki.new("PointLocator")
            self.point_locator.SetDataSet(poly)
            self.point_locator.BuildLocator()

        if kernel == "shepard":
            kern = vtki.new("ShepardKernel")
            kern.SetPowerParameter(2)
        elif kernel == "gaussian":
            kern = vtki.new("GaussianKernel")
        elif kernel == "linear":
            kern = vtki.new("LinearKernel")
        else:
            vedo.logger.error("Error in tovolume(), available kernels are:")
            vedo.logger.error(" [shepard, gaussian, linear]")
            raise RuntimeError()

        if radius:
            kern.SetRadius(radius)

        interpolator = vtki.new("PointInterpolator")
        interpolator.SetInputData(probe)
        interpolator.SetSourceData(poly)
        interpolator.SetKernel(kern)
        interpolator.SetLocator(self.point_locator)

        if n:
            kern.SetNumberOfPoints(n)
            kern.SetKernelFootprintToNClosest()
        else:
            kern.SetRadius(radius)

        if null_value is not None:
            interpolator.SetNullValue(null_value)
        else:
            interpolator.SetNullPointsStrategyToClosestPoint()
        interpolator.Update()

        vol = vedo.Volume(interpolator.GetOutput())

        vol.pipeline = utils.OperationNode(
            "signed_distance",
            parents=[self],
            comment=f"dims={tuple(vol.dimensions())}",
            c="#e9c46a:#0096c7",
        )
        return vol

    def generate_segments(self, istart=0, rmax=1e30, niter=3) -> vedo.shapes.Lines:
        """
        Generate a line segments from a set of points.
        The algorithm is based on the closest point search.

        Returns a `Line` object.
        This object contains the a metadata array of used vertex counts in "UsedVertexCount"
        and the sum of the length of the segments in "SegmentsLengthSum".

        Arguments:
            istart : (int)
                index of the starting point
            rmax : (float)
                maximum length of a segment
            niter : (int)
                number of iterations or passes through the points

        Examples:
            - [moving_least_squares1D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares1D.py)
        """
        points = self.coordinates
        segments = []
        dists = []
        n = len(points)
        used = np.zeros(n, dtype=int)
        for _ in range(niter):
            i = istart
            for _ in range(n):
                p = points[i]
                ids = self.closest_point(p, n=4, return_point_id=True)
                j = ids[1]
                if used[j] > 1 or [j, i] in segments:
                    j = ids[2]
                if used[j] > 1:
                    j = ids[3]
                d = np.linalg.norm(p - points[j])
                if used[j] > 1 or used[i] > 1 or d > rmax:
                    i += 1
                    if i >= n:
                        i = 0
                    continue
                used[i] += 1
                used[j] += 1
                segments.append([i, j])
                dists.append(d)
                i = j
        segments = np.array(segments, dtype=int)

        lines = vedo.shapes.Lines(points[segments], c="k", lw=3)
        lines.metadata["UsedVertexCount"] = used
        lines.metadata["SegmentsLengthSum"] = np.sum(dists)
        lines.pipeline = utils.OperationNode("generate_segments", parents=[self])
        lines.name = "Segments"
        return lines

    def generate_delaunay2d(
        self,
        mode="scipy",
        boundaries=(),
        tol=None,
        alpha=0.0,
        offset=0.0,
        transform=None,
    ) -> vedo.mesh.Mesh:
        """
        Create a mesh from points in the XY plane.
        If `mode='fit'` then the filter computes a best fitting
        plane and projects the points onto it.

        Check also `generate_mesh()`.

        Arguments:
            tol : (float)
                specify a tolerance to control discarding of closely spaced points.
                This tolerance is specified as a fraction of the diagonal length of the bounding box of the points.
            alpha : (float)
                for a non-zero alpha value, only edges or triangles contained
                within a sphere centered at mesh vertices will be output.
                Otherwise, only triangles will be output.
            offset : (float)
                multiplier to control the size of the initial, bounding Delaunay triangulation.
            transform: (LinearTransform, NonLinearTransform)
                a transformation which is applied to points to generate a 2D problem.
                This maps a 3D dataset into a 2D dataset where triangulation can be done on the XY plane.
                The points are transformed and triangulated.
                The topology of triangulated points is used as the output topology.

        Examples:
            - [delaunay2d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/delaunay2d.py)

                ![](https://vedo.embl.es/images/basic/delaunay2d.png)
        """
        plist = self.coordinates.copy()

        #########################################################
        if mode == "scipy":
            from scipy.spatial import Delaunay as scipy_delaunay

            tri = scipy_delaunay(plist[:, 0:2])
            return vedo.mesh.Mesh([plist, tri.simplices])
        ##########################################################

        pd = vtki.vtkPolyData()
        vpts = vtki.vtkPoints()
        vpts.SetData(utils.numpy2vtk(plist, dtype=np.float32))
        pd.SetPoints(vpts)

        delny = vtki.new("Delaunay2D")
        delny.SetInputData(pd)
        if tol:
            delny.SetTolerance(tol)
        delny.SetAlpha(alpha)
        delny.SetOffset(offset)

        if transform:
            delny.SetTransform(transform.T)
        elif mode == "fit":
            delny.SetProjectionPlaneMode(vtki.get_class("VTK_BEST_FITTING_PLANE"))
        elif mode == "xy" and boundaries:
            boundary = vtki.vtkPolyData()
            boundary.SetPoints(vpts)
            cell_array = vtki.vtkCellArray()
            for b in boundaries:
                cpolygon = vtki.vtkPolygon()
                for idd in b:
                    cpolygon.GetPointIds().InsertNextId(idd)
                cell_array.InsertNextCell(cpolygon)
            boundary.SetPolys(cell_array)
            delny.SetSourceData(boundary)

        delny.Update()

        msh = vedo.mesh.Mesh(delny.GetOutput())
        msh.name = "Delaunay2D"
        msh.clean().lighting("off")
        msh.pipeline = utils.OperationNode(
            "delaunay2d",
            parents=[self],
            comment=f"#cells {msh.dataset.GetNumberOfCells()}",
        )
        return msh

    def generate_voronoi(self, padding=0.0, fit=False, method="vtk") -> vedo.Mesh:
        """
        Generate the 2D Voronoi convex tiling of the input points (z is ignored).
        The points are assumed to lie in a plane. The output is a Mesh. Each output cell is a convex polygon.

        A cell array named "VoronoiID" is added to the output Mesh.

        The 2D Voronoi tessellation is a tiling of space, where each Voronoi tile represents the region nearest
        to one of the input points. Voronoi tessellations are important in computational geometry
        (and many other fields), and are the dual of Delaunay triangulations.

        Thus the triangulation is constructed in the x-y plane, and the z coordinate is ignored
        (although carried through to the output).
        If you desire to triangulate in a different plane, you can use fit=True.

        A brief summary is as follows. Each (generating) input point is associated with
        an initial Voronoi tile, which is simply the bounding box of the point set.
        A locator is then used to identify nearby points: each neighbor in turn generates a
        clipping line positioned halfway between the generating point and the neighboring point,
        and orthogonal to the line connecting them. Clips are readily performed by evaluationg the
        vertices of the convex Voronoi tile as being on either side (inside,outside) of the clip line.
        If two intersections of the Voronoi tile are found, the portion of the tile "outside" the clip
        line is discarded, resulting in a new convex, Voronoi tile. As each clip occurs,
        the Voronoi "Flower" error metric (the union of error spheres) is compared to the extent of the region
        containing the neighboring clip points. The clip region (along with the points contained in it) is grown
        by careful expansion (e.g., outward spiraling iterator over all candidate clip points).
        When the Voronoi Flower is contained within the clip region, the algorithm terminates and the Voronoi
        tile is output. Once complete, it is possible to construct the Delaunay triangulation from the Voronoi
        tessellation. Note that topological and geometric information is used to generate a valid triangulation
        (e.g., merging points and validating topology).

        Arguments:
            pts : (list)
                list of input points.
            padding : (float)
                padding distance. The default is 0.
            fit : (bool)
                detect automatically the best fitting plane. The default is False.

        Examples:
            - [voronoi1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/voronoi1.py)

                ![](https://vedo.embl.es/images/basic/voronoi1.png)

            - [voronoi2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/voronoi2.py)

                ![](https://vedo.embl.es/images/advanced/voronoi2.png)
        """
        pts = self.coordinates

        if method == "scipy":
            from scipy.spatial import Voronoi as scipy_voronoi

            pts = np.asarray(pts)[:, (0, 1)]
            vor = scipy_voronoi(pts)
            regs = []  # filter out invalid indices
            for r in vor.regions:
                flag = True
                for x in r:
                    if x < 0:
                        flag = False
                        break
                if flag and len(r) > 0:
                    regs.append(r)

            m = vedo.Mesh([vor.vertices, regs])
            m.celldata["VoronoiID"] = np.array(list(range(len(regs)))).astype(int)

        elif method == "vtk":
            vor = vtki.new("Voronoi2D")
            if isinstance(pts, vedo.pointcloud.Points):
                vor.SetInputData(pts.dataset)
            else:
                pts = np.asarray(pts)
                if pts.shape[1] == 2:
                    pts = np.c_[pts, np.zeros(len(pts))]
                pd = vtki.vtkPolyData()
                vpts = vtki.vtkPoints()
                vpts.SetData(utils.numpy2vtk(pts, dtype=np.float32))
                pd.SetPoints(vpts)
                vor.SetInputData(pd)
            vor.SetPadding(padding)
            vor.SetGenerateScalarsToPointIds()
            if fit:
                vor.SetProjectionPlaneModeToBestFittingPlane()
            else:
                vor.SetProjectionPlaneModeToXYPlane()
            vor.Update()
            poly = vor.GetOutput()
            arr = poly.GetCellData().GetArray(0)
            if arr:
                arr.SetName("VoronoiID")
            m = vedo.Mesh(poly, c="orange5")

        else:
            vedo.logger.error(f"Unknown method {method} in voronoi()")
            raise RuntimeError

        m.lw(2).lighting("off").wireframe()
        m.name = "Voronoi"
        return m

    def generate_delaunay3d(self, radius=0, tol=None) -> vedo.TetMesh:
        """
        Create 3D Delaunay triangulation of input points.

        Arguments:
            radius : (float)
                specify distance (or "alpha") value to control output.
                For a non-zero values, only tetra contained within the circumsphere
                will be output.
            tol : (float)
                Specify a tolerance to control discarding of closely spaced points.
                This tolerance is specified as a fraction of the diagonal length of
                the bounding box of the points.
        """
        deln = vtki.new("Delaunay3D")
        deln.SetInputData(self.dataset)
        deln.SetAlpha(radius)
        deln.AlphaTetsOn()
        deln.AlphaTrisOff()
        deln.AlphaLinesOff()
        deln.AlphaVertsOff()
        deln.BoundingTriangulationOff()
        if tol:
            deln.SetTolerance(tol)
        deln.Update()
        m = vedo.TetMesh(deln.GetOutput())
        m.pipeline = utils.OperationNode(
            "generate_delaunay3d", c="#e9c46a:#edabab", parents=[self],
        )
        m.name = "Delaunay3D"
        return m
