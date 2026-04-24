#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Common algorithm mixins shared by vedo objects."""

from typing import Any
from typing_extensions import Self
import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import utils
from vedo.core.transformations import LinearTransform
from vedo.core.data import DataArrayHelper, _get_data_legacy_format

__all__ = ["CommonAlgorithms"]


class CommonAlgorithms:
    """Common algorithms."""

    def _ensure_cell_locator(self):
        """Build and cache a cell locator for the current dataset if missing."""
        if not self.cell_locator:
            self.cell_locator = vtki.new("CellTreeLocator")
            self.cell_locator.SetDataSet(self.dataset)
            self.cell_locator.BuildLocator()
        return self.cell_locator

    @staticmethod
    def _vtk_idlist_to_numpy(id_list) -> np.ndarray:
        """Convert a vtkIdList to a numpy array of ids."""
        return np.array([id_list.GetId(i) for i in range(id_list.GetNumberOfIds())])

    @staticmethod
    def _parse_vtk_flat_connectivity(arr1d) -> list:
        """Unpack VTK flat connectivity [nids, id0...idn, ...] into a list of lists."""
        i = 0
        conn = []
        n = len(arr1d)
        while i < n:
            nids = arr1d[i]
            conn.append([int(arr1d[i + k]) for k in range(1, nids + 1)])
            i += nids + 1
        return conn

    def _run_gradient_filter(
        self,
        mode: str,
        on: str,
        array_name: str | None,
        fast: bool,
        result_name: str,
    ) -> np.ndarray:
        """Shared implementation for gradient/divergence/vorticity extraction."""
        gf = vtki.new("GradientFilter")
        if on.startswith("p"):
            varr = self.dataset.GetPointData()
            assoc = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS
            def getter(out):
                return out.GetPointData().GetArray(result_name)
        elif on.startswith("c"):
            varr = self.dataset.GetCellData()
            assoc = vtki.vtkDataObject.FIELD_ASSOCIATION_CELLS
            def getter(out):
                return out.GetCellData().GetArray(result_name)
        else:
            vedo.logger.error(f"in {mode}(): unknown option {on}")
            raise RuntimeError(f"in {mode}(): unknown option '{on}'")

        if array_name is None:
            if mode == "gradient":
                active = varr.GetScalars()
                if not active:
                    vedo.logger.error(f"in gradient: no scalars found for {on}")
                    raise RuntimeError(f"in gradient(): no scalars found for '{on}'")
            else:
                active = varr.GetVectors()
                if not active:
                    vedo.logger.error(f"in {mode}(): no vectors found for {on}")
                    raise RuntimeError(f"in {mode}(): no vectors found for '{on}'")
            array_name = active.GetName()

        gf.SetInputData(self.dataset)
        gf.SetInputScalars(assoc, array_name)
        gf.SetFasterApproximation(fast)

        if mode == "gradient":
            gf.ComputeGradientOn()
        else:
            gf.ComputeGradientOff()
        if mode == "divergence":
            gf.ComputeDivergenceOn()
        else:
            gf.ComputeDivergenceOff()
        if mode == "vorticity":
            gf.ComputeVorticityOn()
        else:
            gf.ComputeVorticityOff()

        if mode == "gradient":
            gf.SetResultArrayName(result_name)
        elif mode == "divergence":
            gf.SetDivergenceArrayName(result_name)
        elif mode == "vorticity":
            gf.SetVorticityArrayName(result_name)

        gf.Update()
        return utils.vtk2numpy(getter(gf.GetOutput()))

    # ====== Data access ======

    @property
    def pointdata(self):
        """
        Return a `DataArrayHelper` to access point (vertex) data arrays.
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.pointdata["arrayname"]`

        Usage:
            - `myobj.pointdata.keys()` returns the available data array names.
            - `myobj.pointdata.select(name)` makes this array the active one.
            - `myobj.pointdata.remove(name)` removes this array.
        """
        return DataArrayHelper(self, 0)

    @property
    def celldata(self):
        """
        Return a `DataArrayHelper` to access cell (face) data arrays.
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.celldata["arrayname"]`

        Usage:
            - `myobj.celldata.keys()` returns the available data array names.
            - `myobj.celldata.select(name)` makes this array the active one.
            - `myobj.celldata.remove(name)` removes this array.
        """
        return DataArrayHelper(self, 1)

    @property
    def metadata(self):
        """
        Return a `DataArrayHelper` to access field data arrays (not tied to points or cells).
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.metadata["arrayname"]`

        Usage:
            - `myobj.metadata.keys()` returns the available data array names.
            - `myobj.metadata.select(name)` makes this array the active one.
            - `myobj.metadata.remove(name)` removes this array.
        """
        return DataArrayHelper(self, 2)

    # ====== Object info ======

    def rename(self, newname: str) -> Self:
        """Rename the object"""
        try:
            self.name = newname
        except AttributeError:
            vedo.logger.error(f"Cannot rename object {self}")
        return self

    def memory_address(self) -> int:
        """
        Return a unique memory address integer which may serve as the ID of the
        object, or passed to c++ code.
        """
        # https://www.linkedin.com/pulse/speedup-your-code-accessing-python-vtk-objects-from-c-pletzer/
        # https://github.com/tfmoraes/polydata_connectivity
        return int(self.dataset.GetAddressAsString("")[5:], 16)

    def memory_size(self) -> int:
        """Return the approximate memory size of the object in kilobytes."""
        return self.dataset.GetActualMemorySize()

    def modified(self) -> Self:
        """Use in conjunction with `tonumpy()` to update any modifications to the image array."""
        self.dataset.GetPointData().Modified()
        scals = self.dataset.GetPointData().GetScalars()
        if scals:
            scals.Modified()
        return self

    def box(self, scale=1, padding=0) -> vedo.Mesh:
        """
        Return the bounding box as a new `Mesh` object.

        Args:
            scale (float):
                box size can be scaled by a factor
            padding (float, list):
                a constant padding can be added (can be a list `[padx,pady,padz]`)
        """
        b = self.bounds()
        if not utils.is_sequence(padding):
            padding = [padding, padding, padding]
        length, width, height = b[1] - b[0], b[3] - b[2], b[5] - b[4]
        tol = (length + width + height) / 30000  # useful for boxing text
        pos = [(b[0] + b[1]) / 2, (b[3] + b[2]) / 2, (b[5] + b[4]) / 2 - tol]
        bx = vedo.shapes.Box(
            pos,
            length * scale + padding[0],
            width * scale + padding[1],
            height * scale + padding[2],
            c="gray",
        )
        try:
            pr = vtki.vtkProperty()
            pr.DeepCopy(self.properties)
            bx.actor.SetProperty(pr)
            bx.properties = pr
        except (AttributeError, TypeError):
            pass
        bx.flat().lighting("off").wireframe(True)
        return bx

    def update_dataset(self, dataset, **kwargs) -> Self:
        """Update the dataset of the object with the provided VTK dataset."""
        self._update(dataset, **kwargs)
        return self

    # ====== Geometry & bounds ======

    def bounds(self) -> np.ndarray:
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        try:  # this is very slow for large meshes
            pts = self.vertices
            xmin, ymin, zmin = np.nanmin(pts, axis=0)
            xmax, ymax, zmax = np.nanmax(pts, axis=0)
            return np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        except (AttributeError, ValueError):
            return np.array(self.dataset.GetBounds())

    def xbounds(self) -> np.ndarray:
        """Get the bounds `[xmin,xmax]`."""
        b = self.bounds()
        return np.array([b[0], b[1]])

    def ybounds(self) -> np.ndarray:
        """Get the bounds `[ymin,ymax]`."""
        b = self.bounds()
        return np.array([b[2], b[3]])

    def zbounds(self) -> np.ndarray:
        """Get the bounds `[zmin,zmax]`."""
        b = self.bounds()
        return np.array([b[4], b[5]])

    def diagonal_size(self) -> float:
        """Get the length of the diagonal of the bounding box."""
        b = self.bounds()
        return np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)

    def average_size(self) -> float:
        """
        Calculate and return the average size of the object.
        This is the mean of the vertex distances from the center of mass.
        """
        coords = self.vertices
        if coords.shape[0] == 0:
            return 0.0
        cm = np.mean(coords, axis=0)
        cc = coords - cm
        return np.mean(np.linalg.norm(cc, axis=1))

    def center_of_mass(self) -> np.ndarray:
        """Get the center of mass of the object."""
        if isinstance(self, (vedo.RectilinearGrid, vedo.Volume)):
            return np.array(self.dataset.GetCenter())
        cmf = vtki.new("CenterOfMass")
        cmf.SetInputData(self.dataset)
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)

    def copy_data_from(self, obj: Any) -> Self:
        """Copy all data (point and cell data) from this input object"""
        self.dataset.GetPointData().PassData(obj.dataset.GetPointData())
        self.dataset.GetCellData().PassData(obj.dataset.GetCellData())
        self.pipeline = utils.OperationNode(
            "copy_data_from",
            parents=[self, obj],
            comment=f"{obj.__class__.__name__}",
            shape="note",
            c="#ccc5b9",
        )
        return self

    def inputdata(self):
        """Obsolete, use `.dataset` instead."""
        vedo.logger.warning(
            "'inputdata()' is obsolete, use '.dataset' instead."
        )
        return self.dataset

    @property
    def npoints(self):
        """Retrieve the number of points (or vertices)."""
        return self.dataset.GetNumberOfPoints()

    @property
    def nvertices(self):
        """Retrieve the number of vertices (or points)."""
        return self.dataset.GetNumberOfPoints()

    @property
    def ncells(self):
        """Retrieve the number of cells."""
        return self.dataset.GetNumberOfCells()

    def cell_centers(self, copy_arrays=False) -> vedo.Points:
        """
        Get the coordinates of the cell centers as a `Points` object.

        Examples:
            - [delaunay2d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/delaunay2d.py)
        """
        vcen = vtki.new("CellCenters")
        vcen.SetCopyArrays(copy_arrays)
        vcen.SetVertexCells(copy_arrays)
        vcen.SetInputData(self.dataset)
        vcen.Update()
        vpts = vedo.Points(vcen.GetOutput())
        if copy_arrays:
            vpts.copy_properties_from(self)
        return vpts

    # ====== Connectivity ======

    @property
    def lines(self):
        """
        Get lines connectivity ids as a python array
        formatted as `[[id0,id1], [id3,id4], ...]`

        See also: `lines_as_flat_array()`.
        """
        try:
            arr1d = _get_data_legacy_format(self.dataset.GetLines())
        except AttributeError:
            return []
        return self._parse_vtk_flat_connectivity(arr1d)

    @property
    def lines_as_flat_array(self):
        """
        Get lines connectivity ids as a 1D numpy array.
        Format is e.g. [2,  10,20,  3, 10,11,12,  2, 70,80, ...]

        See also: `lines()`.
        """
        try:
            return _get_data_legacy_format(self.dataset.GetLines())
        except AttributeError:
            return np.array([], dtype=int)

    # ====== Spatial queries ======

    def mark_boundaries(self) -> Self:
        """
        Mark cells and vertices if they lie on a boundary.
        A new array called `BoundaryCells` is added to the object.
        """
        mb = vtki.new("MarkBoundaryFilter")
        mb.SetInputData(self.dataset)
        mb.Update()
        self.dataset.DeepCopy(mb.GetOutput())
        self.pipeline = utils.OperationNode("mark_boundaries", parents=[self])
        return self

    def find_cells_in_bounds(self, xbounds=(), ybounds=(), zbounds=()) -> np.ndarray:
        """
        Find cells that are within the specified bounds.
        """
        try:
            xbounds = list(xbounds.bounds())
        except AttributeError:
            pass

        if len(xbounds) == 6:
            bnds = xbounds
        else:
            bnds = list(self.bounds())
            if len(xbounds) == 2:
                bnds[0] = xbounds[0]
                bnds[1] = xbounds[1]
            if len(ybounds) == 2:
                bnds[2] = ybounds[0]
                bnds[3] = ybounds[1]
            if len(zbounds) == 2:
                bnds[4] = zbounds[0]
                bnds[5] = zbounds[1]

        cell_ids = vtki.vtkIdList()
        self._ensure_cell_locator().FindCellsWithinBounds(bnds, cell_ids)
        return self._vtk_idlist_to_numpy(cell_ids)

    def find_cells_along_line(self, p0, p1, tol=0.001) -> np.ndarray:
        """
        Find cells that are intersected by a line segment.
        """
        cell_ids = vtki.vtkIdList()
        self._ensure_cell_locator().FindCellsAlongLine(p0, p1, tol, cell_ids)
        return self._vtk_idlist_to_numpy(cell_ids)

    def find_cells_along_plane(self, origin, normal, tol=0.001) -> np.ndarray:
        """
        Find cells that are intersected by a plane.
        """
        cell_ids = vtki.vtkIdList()
        self._ensure_cell_locator().FindCellsAlongPlane(origin, normal, tol, cell_ids)
        return self._vtk_idlist_to_numpy(cell_ids)

    def keep_cell_types(self, types=()) -> Self:
        """
        Extract cells of a specific type.

        Check the VTK cell types here:
        https://vtk.org/doc/nightly/html/vtkCellType_8h.html
        """
        fe = vtki.new("ExtractCellsByType")
        fe.SetInputData(self.dataset)
        for t in types:
            try:
                if utils.is_integer(t):
                    it = t
                else:
                    it = vtki.cell_types[t.upper()]
            except KeyError:
                vedo.logger.error(f"Cell type '{t}' not recognized")
                continue
            fe.AddCellType(it)
        fe.Update()
        self._update(fe.GetOutput())
        return self

    # ====== Data mapping ======

    def map_cells_to_points(self, arrays=(), move=False) -> Self:
        """
        Interpolate cell data (i.e., data specified per cell or face)
        into point data (i.e., data specified at each vertex).
        The method of transformation is based on averaging the data values
        of all cells using a particular point.

        A custom list of arrays to be mapped can be passed in input.

        Set `move=True` to delete the original `celldata` array.
        """
        c2p = vtki.new("CellDataToPointData")
        c2p.SetInputData(self.dataset)
        if not move:
            c2p.PassCellDataOn()
        if arrays:
            c2p.ClearCellDataArrays()
            c2p.ProcessAllArraysOff()
            for arr in arrays:
                c2p.AddCellDataArray(arr)
        else:
            c2p.ProcessAllArraysOn()
        c2p.Update()
        self._update(c2p.GetOutput(), reset_locators=False)
        self.mapper.SetScalarModeToUsePointData()
        self.pipeline = utils.OperationNode("map_cells_to_points", parents=[self])
        return self

    # ====== Vertices & coordinates ======

    @property
    def vertices(self):
        """
        Return the vertices (points) coordinates.
        This is equivalent to `points` and `coordinates`.
        """
        try:
            # for polydata and unstructured grid
            vpts = self.dataset.GetPoints()
            if vpts is None:
                return np.array([], dtype=float)
            varr = vpts.GetData()
        except AttributeError:
            # 'vtkImageData' object has no attribute 'GetPoints'
            v2p = vtki.new("ImageToPoints")
            v2p.SetInputData(self.dataset)
            v2p.Update()
            varr = v2p.GetOutput().GetPoints().GetData()
        except TypeError:
            # for RectilinearGrid, StructuredGrid
            vpts = vtki.vtkPoints()
            self.dataset.GetPoints(vpts)
            varr = vpts.GetData()
        except Exception as e:
            vedo.logger.error(f"Cannot get point coords for {type(self)}: {e}")
            return np.array([], dtype=float)

        return utils.vtk2numpy(varr)

    # setter
    @vertices.setter
    def vertices(self, pts):
        """Set vertex coordinates. Same as `points` and `coordinates`."""
        pts = utils.make3d(pts)
        arr = utils.numpy2vtk(pts, dtype=np.float32)
        try:
            vpts = self.dataset.GetPoints()
            vpts.SetData(arr)
            vpts.Modified()
        except (AttributeError, TypeError):
            vedo.logger.error(f"Cannot set vertices for {type(self)}")
            return
        # reset mesh to identity matrix position/rotation:
        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        self.transform = LinearTransform()

    @property
    def points(self):
        """
        Return the points coordinates. Same as `vertices` and `coordinates`.
        """
        return self.vertices

    @points.setter
    def points(self, pts):
        """Set points coordinates. Same as `vertices` and `coordinates`."""
        self.vertices = pts

    @property
    def coordinates(self):
        """Return the points coordinates. Same as `vertices` and `points`."""
        return self.vertices

    @coordinates.setter
    def coordinates(self, pts):
        """Set points coordinates. Same as `vertices` and `points`."""
        self.vertices = pts

    # ====== Cell connectivity ======

    @property
    def cells_as_flat_array(self):
        """
        Get cell connectivity ids as a 1D numpy array.
        Format is e.g. [3,  10,20,30  4, 10,11,12,13  ...]
        """
        try:
            # valid for unstructured grid
            arr1d = _get_data_legacy_format(self.dataset.GetCells())
        except AttributeError:
            try:
                # valid for polydata
                arr1d = _get_data_legacy_format(self.dataset.GetPolys())
            except AttributeError:
                return np.array([], dtype=int)
        return arr1d

    @property
    def cells(self):
        """
        Get the cells connectivity ids as a numpy array.

        The output format is: `[[id0 ... idn], [id0 ... idm],  etc]`.
        """
        try:
            # valid for unstructured grid
            arr1d = _get_data_legacy_format(self.dataset.GetCells())
        except AttributeError:
            try:
                # valid for polydata
                arr1d = _get_data_legacy_format(self.dataset.GetPolys())
            except AttributeError:
                return []
        return self._parse_vtk_flat_connectivity(arr1d)

    def cell_edge_neighbors(self):
        """
        Get the cell neighbor indices of each cell.

        Returns a python list of lists.
        """

        def face_to_edges(face):
            n = len(face)
            return [[face[i], face[(i + 1) % n]] for i in range(n)]

        pd = self.dataset
        pd.BuildLinks()

        neicells = []
        for i, cell in enumerate(self.cells):
            nn = []
            for edge in face_to_edges(cell):
                neighbors = vtki.vtkIdList()
                pd.GetCellEdgeNeighbors(i, edge[0], edge[1], neighbors)
                if neighbors.GetNumberOfIds() > 0:
                    neighbor = neighbors.GetId(0)
                    nn.append(neighbor)
            neicells.append(nn)

        return neicells

    def map_points_to_cells(self, arrays=(), move=False) -> Self:
        """
        Interpolate point data (i.e., data specified per point or vertex)
        into cell data (i.e., data specified per cell).
        The method of transformation is based on averaging the data values
        of all points defining a particular cell.

        A custom list of arrays to be mapped can be passed in input.

        Set `move=True` to delete the original `pointdata` array.

        Examples:
            - [mesh_map2cell.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_map2cell.py)
        """
        p2c = vtki.new("PointDataToCellData")
        p2c.SetInputData(self.dataset)
        if not move:
            p2c.PassPointDataOn()
        if arrays:
            p2c.ClearPointDataArrays()
            p2c.ProcessAllArraysOff()
            for arr in arrays:
                p2c.AddPointDataArray(arr)
        else:
            p2c.ProcessAllArraysOn()
        p2c.Update()
        self._update(p2c.GetOutput(), reset_locators=False)
        self.mapper.SetScalarModeToUseCellData()
        self.pipeline = utils.OperationNode("map_points_to_cells", parents=[self])
        return self

    def resample_data_from(self, source, tol=None, categorical=False) -> Self:
        """
        Resample point and cell data from another dataset.
        The output has the same structure but its point data have
        the resampled values from target.

        Use `tol` to set the tolerance used to compute whether
        a point in the source is in a cell of the current object.
        Points without resampled values, and their cells, are marked as blank.
        If the data is categorical, then the resulting data will be determined
        by a nearest neighbor interpolation scheme.

        Examples:
        ```python
        from vedo import *
        m1 = Mesh(dataurl+'bunny.obj')#.add_gaussian_noise(0.1)
        pts = m1.coordinates
        ces = m1.cell_centers().coordinates
        m1.pointdata["xvalues"] = np.power(pts[:,0], 3)
        m1.celldata["yvalues"]  = np.power(ces[:,1], 3)
        m2 = Mesh(dataurl+'bunny.obj')
        m2.resample_data_from(m1)
        # print(m2.pointdata["xvalues"])
        show(m1, m2 , N=2, axes=1)
        ```
        """
        rs = vtki.new("ResampleWithDataSet")
        rs.SetInputData(self.dataset)
        rs.SetSourceData(source.dataset)

        rs.SetPassPointArrays(True)
        rs.SetPassCellArrays(True)
        rs.SetPassFieldArrays(True)
        rs.SetCategoricalData(categorical)

        rs.SetComputeTolerance(True)
        if tol is not None:
            rs.SetComputeTolerance(False)
            rs.SetTolerance(tol)
        rs.Update()
        self._update(rs.GetOutput(), reset_locators=False)
        self.pipeline = utils.OperationNode(
            "resample_data_from",
            comment=f"{source.__class__.__name__}",
            parents=[self, source],
        )
        return self

    def interpolate_data_from(
        self,
        source,
        radius=None,
        n=None,
        kernel="shepard",
        exclude=("Normals",),
        on="points",
        null_strategy=1,
        null_value=0,
    ) -> Self:
        """
        Interpolate over source to port its data onto the current object using various kernels.

        If n (number of closest points to use) is set then radius value is ignored.

        Check out also:
            `probe()` which in many cases can be faster.

        Args:
            kernel (str):
                available kernels are [shepard, gaussian, linear]
            null_strategy (int):
                specify a strategy to use when encountering a "null" point
                during the interpolation process. Null points occur when the local neighborhood
                (of nearby points to interpolate from) is empty.

                - Case 0: an output array is created that marks points
                  as being valid (=1) or null (invalid =0), and the null_value is set as well
                - Case 1: the output data value(s) are set to the provided null_value
                - Case 2: simply use the closest point to perform the interpolation.
            null_value (float):
                see above.

        Examples:
            - [interpolate_scalar1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolate_scalar1.py)
            - [interpolate_scalar3.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolate_scalar3.py)
            - [interpolate_scalar4.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolate_scalar4.py)
            - [image_probe.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/image_probe.py)

                ![](https://vedo.embl.es/images/advanced/interpolateMeshArray.png)
        """
        if radius is None and not n:
            vedo.logger.error(
                "in interpolate_data_from(): please set either radius or n"
            )
            raise RuntimeError("in interpolate_data_from(): please set either radius or n")

        if on == "points":
            points = source.dataset
        elif on == "cells":
            c2p = vtki.new("CellDataToPointData")
            c2p.SetInputData(source.dataset)
            c2p.Update()
            points = c2p.GetOutput()
        else:
            vedo.logger.error(
                "in interpolate_data_from(), on must be on points or cells"
            )
            raise RuntimeError("in interpolate_data_from(): 'on' must be 'points' or 'cells'")

        locator = vtki.new("PointLocator")
        locator.SetDataSet(points)
        locator.BuildLocator()

        if kernel.lower() == "shepard":
            kern = vtki.new("ShepardKernel")
            kern.SetPowerParameter(2)
        elif kernel.lower() == "gaussian":
            kern = vtki.new("GaussianKernel")
            kern.SetSharpness(2)
        elif kernel.lower() == "linear":
            kern = vtki.new("LinearKernel")
        else:
            vedo.logger.error("available kernels are: [shepard, gaussian, linear]")
            raise RuntimeError("in interpolate_data_from(): unknown kernel, use shepard/gaussian/linear")

        if n:
            kern.SetNumberOfPoints(n)
            kern.SetKernelFootprintToNClosest()
        else:
            kern.SetRadius(radius)
            kern.SetKernelFootprintToRadius()

        # remove arrays already present in self so the interpolator doesn't skip them
        clspd = self.dataset.GetPointData()
        clsnames = [clspd.GetArrayName(i) for i in range(clspd.GetNumberOfArrays())]
        srcpd = points.GetPointData()
        pointsnames = [srcpd.GetArrayName(i) for i in range(srcpd.GetNumberOfArrays())]

        for cname in clsnames:
            if cname in set(pointsnames) - set(exclude):
                clspd.RemoveArray(cname)

        interpolator = vtki.new("PointInterpolator")
        interpolator.SetInputData(self.dataset)
        interpolator.SetSourceData(points)
        interpolator.SetKernel(kern)
        interpolator.SetLocator(locator)
        interpolator.PassFieldArraysOn()
        interpolator.SetNullPointsStrategy(null_strategy)
        interpolator.SetNullValue(null_value)
        interpolator.SetValidPointsMaskArrayName("ValidPointMask")
        for ex in exclude:
            interpolator.AddExcludedArray(ex)

        # Keep existing non-overlapping arrays on destination dataset.
        # Overlapping names were already removed above to force interpolation output.

        interpolator.Update()

        if on == "cells":
            p2c = vtki.new("PointDataToCellData")
            p2c.SetInputData(interpolator.GetOutput())
            p2c.Update()
            cpoly = p2c.GetOutput()
        else:
            cpoly = interpolator.GetOutput()

        self._update(cpoly, reset_locators=False)

        self.pipeline = utils.OperationNode(
            "interpolate_data_from", parents=[self, source]
        )
        return self

    def add_ids(self) -> Self:
        """
        Generate point and cell ids arrays.

        Two new arrays are added to the mesh named `PointID` and `CellID`.
        """
        ids = vtki.new_ids_filter()
        if ids is None:
            vedo.logger.error(
                "add_ids(): cannot instantiate vtkIdFilter/vtkGenerateIds"
            )
            raise RuntimeError("add_ids(): missing VTK ids filter")
        ids.SetInputData(self.dataset)
        ids.PointIdsOn()
        ids.CellIdsOn()
        ids.FieldDataOff()
        ids.SetPointIdsArrayName("PointID")
        ids.SetCellIdsArrayName("CellID")
        ids.Update()
        # self._update(ids.GetOutput(), reset_locators=False)  # https://github.com/marcomusy/vedo/issues/1267
        point_arr = ids.GetOutput().GetPointData().GetArray("PointID")
        cell_arr = ids.GetOutput().GetCellData().GetArray("CellID")
        if point_arr:
            self.dataset.GetPointData().AddArray(point_arr)
        if cell_arr:
            self.dataset.GetCellData().AddArray(cell_arr)
        self.pipeline = utils.OperationNode("add_ids", parents=[self])
        return self

    # ====== Field operations ======

    def gradient(self, input_array=None, on="points", fast=False) -> np.ndarray:
        """
        Compute and return the gradient of the active scalar field as a numpy array.

        Args:
            input_array (str):
                array of the scalars to compute the gradient,
                if None the current active array is selected
            on (str):
                compute either on 'points' or 'cells' data
            fast (bool):
                if True, will use a less accurate algorithm
                that performs fewer derivative calculations (and is therefore faster).

        Examples:
            - [isolines.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/isolines.py)

            ![](https://user-images.githubusercontent.com/32848391/72433087-f00a8780-3798-11ea-9778-991f0abeca70.png)
        """
        return self._run_gradient_filter(
            mode="gradient",
            on=on,
            array_name=input_array,
            fast=fast,
            result_name="Gradient",
        )

    def divergence(self, array_name=None, on="points", fast=False) -> np.ndarray:
        """
        Compute and return the divergence of a vector field as a numpy array.

        Args:
            array_name (str):
                name of the array of vectors to compute the divergence,
                if None the current active array is selected
            on (str):
                compute either on 'points' or 'cells' data
            fast (bool):
                if True, will use a less accurate algorithm
                that performs fewer derivative calculations and is therefore faster.
        """
        return self._run_gradient_filter(
            mode="divergence",
            on=on,
            array_name=array_name,
            fast=fast,
            result_name="Divergence",
        )

    def vorticity(self, array_name=None, on="points", fast=False) -> np.ndarray:
        """
        Compute and return the vorticity of a vector field as a numpy array.

        Args:
            array_name (str):
                name of the array to compute the vorticity,
                if None the current active array is selected
            on (str):
                compute either on 'points' or 'cells' data
            fast (bool):
                if True, will use a less accurate algorithm
                that performs fewer derivative calculations (and is therefore faster).
        """
        return self._run_gradient_filter(
            mode="vorticity",
            on=on,
            array_name=array_name,
            fast=fast,
            result_name="Vorticity",
        )

    def probe(
        self,
        source,
        categorical=False,
        snap=False,
        tol=0,
    ) -> Self:
        """
        Takes a data set and probes its scalars at the specified points in space.

        Note that a mask is also output with valid/invalid points which can be accessed
        with `mesh.pointdata['ValidPointMask']`.

        Args:
            source : any dataset
                the data set to probe.
            categorical : bool
                control whether the source pointdata is to be treated as categorical.
            snap : bool
                snap to the cell with the closest point if no cell was found
            tol : float
                the tolerance to use when performing the probe.

        Check out also:
            `interpolate_data_from()` and `tovolume()`

        Examples:
            - [probe_points.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/probe_points.py)

                ![](https://vedo.embl.es/images/volumetric/probePoints.png)
        """
        probe_filter = vtki.new("ProbeFilter")
        probe_filter.SetSourceData(source.dataset)
        probe_filter.SetInputData(self.dataset)
        probe_filter.PassCellArraysOn()
        probe_filter.PassFieldArraysOn()
        probe_filter.PassPointArraysOn()
        probe_filter.SetCategoricalData(categorical)
        probe_filter.ComputeToleranceOff()
        if tol:
            probe_filter.ComputeToleranceOn()
            probe_filter.SetTolerance(tol)
        probe_filter.SetSnapToCellWithClosestPoint(snap)
        probe_filter.Update()
        self._update(probe_filter.GetOutput(), reset_locators=False)
        self.pipeline = utils.OperationNode("probe", parents=[self, source])
        self.pointdata.rename("vtkValidPointMask", "ValidPointMask")
        return self

    def compute_cell_size(self) -> Self:
        """
        Add to this object a cell data array
        containing the area, volume and edge length
        of the cells (when applicable to the object type).

        Array names are: `Area`, `Volume`, `Length`.
        """
        csf = vtki.new("CellSizeFilter")
        csf.SetInputData(self.dataset)
        csf.SetComputeArea(1)
        csf.SetComputeVolume(1)
        csf.SetComputeLength(1)
        csf.SetComputeVertexCount(0)
        csf.SetAreaArrayName("Area")
        csf.SetVolumeArrayName("Volume")
        csf.SetLengthArrayName("Length")
        csf.Update()
        self._update(csf.GetOutput(), reset_locators=False)
        self.pipeline = utils.OperationNode("compute_cell_size", parents=[self])
        return self

    def generate_random_data(self) -> Self:
        """Fill a dataset with random attributes"""
        gen = vtki.new("RandomAttributeGenerator")
        gen.SetInputData(self.dataset)
        gen.GenerateAllDataOn()
        gen.SetDataTypeToFloat()
        gen.GeneratePointNormalsOff()
        gen.GeneratePointTensorsOn()
        gen.GenerateCellScalarsOn()
        gen.Update()
        self._update(gen.GetOutput(), reset_locators=False)
        self.pipeline = utils.OperationNode("generate_random_data", parents=[self])
        return self

    def integrate_data(self) -> dict:
        """
        Integrate point and cell data arrays while computing length,
        area or volume of the domain. It works for 1D, 2D or 3D cells.

        For volumetric datasets, this filter ignores all but 3D cells.
        It will not compute the volume contained in a closed surface.

        Returns a dictionary with keys: `pointdata`, `celldata`, `metadata`,
        which contain the integration result for the corresponding attributes.

        Examples:
            ```python
            from vedo import *
            surf = Sphere(res=100)
            surf.pointdata['scalars'] = np.ones(surf.npoints)
            data = surf.integrate_data()
            print(data['pointdata']['scalars'], "is equal to 4pi", 4*np.pi)
            ```

            ```python
            from vedo import *

            xcoords1 = np.arange(0, 2.2, 0.2)
            xcoords2 = sqrt(np.arange(0, 4.2, 0.2))

            ycoords = np.arange(0, 1.2, 0.2)

            surf1 = Grid(s=(xcoords1, ycoords)).rotate_y(-45).lw(2)
            surf2 = Grid(s=(xcoords2, ycoords)).rotate_y(-45).lw(2)

            surf1.pointdata['scalars'] = surf1.vertices[:,2]
            surf2.pointdata['scalars'] = surf2.vertices[:,2]

            data1 = surf1.integrate_data()
            data2 = surf2.integrate_data()

            print(data1['pointdata']['scalars'],
                "is equal to",
                data2['pointdata']['scalars'],
                "even if the grids are different!",
                "(= the volume under the surface)"
            )
            show(surf1, surf2, N=2, axes=1).close()
            ```
        """
        vinteg = vtki.new("IntegrateAttributes")
        vinteg.SetInputData(self.dataset)
        vinteg.Update()
        ugrid = vedo.UnstructuredGrid(vinteg.GetOutput())
        data = dict(
            pointdata=ugrid.pointdata.todict(),
            celldata=ugrid.celldata.todict(),
            metadata=ugrid.metadata.todict(),
        )
        return data

    # ====== IO & conversion ======

    def write(self, filename, binary=True) -> None:
        """Write object to file."""
        vedo.file_io.write(self, filename, binary)
        self.pipeline = utils.OperationNode(
            "write",
            parents=[self],
            comment=str(filename)[:15],
            shape="folder",
            c="#8a817c",
        )

    def tomesh(self, bounds=(), shrink=0) -> vedo.Mesh:
        """
        Extract boundary geometry from dataset (or convert data to polygonal type).

        Two new arrays are added to the mesh: `OriginalCellIds` and `OriginalPointIds`
        to keep track of the original mesh elements.

        Args:
            bounds (list):
                specify a sub-region to extract
            shrink (float):
                shrink the cells to a fraction of their original size
        """
        geo = vtki.new("GeometryFilter")

        if shrink:
            sf = vtki.new("ShrinkFilter")
            sf.SetInputData(self.dataset)
            sf.SetShrinkFactor(shrink)
            sf.Update()
            geo.SetInputData(sf.GetOutput())
        else:
            geo.SetInputData(self.dataset)

        geo.SetPassThroughCellIds(1)
        geo.SetPassThroughPointIds(1)
        geo.SetOriginalCellIdsName("OriginalCellIds")
        geo.SetOriginalPointIdsName("OriginalPointIds")
        geo.SetNonlinearSubdivisionLevel(1)
        # geo.MergingOff() # crashes on StructuredGrids
        if bounds:
            geo.SetExtent(bounds)
            geo.ExtentClippingOn()
        geo.Update()
        msh = vedo.mesh.Mesh(geo.GetOutput())
        msh.pipeline = utils.OperationNode("tomesh", parents=[self], c="#9e2a2b")
        return msh

    # ====== Distance operations ======

    def signed_distance(
        self, dims=(20, 20, 20), bounds=None, invert=False, max_radius=None
    ) -> vedo.Volume:
        """
        Compute the `Volume` object whose voxels contains the signed distance from
        the object. The calling object must have "Normals" defined.

        Args:
            bounds (list, actor):
                bounding box sizes
            dims (list):
                dimensions (nr. of voxels) of the output volume.
            invert (bool):
                flip the sign
            max_radius (float):
                specify how far out to propagate distance calculation

        Examples:
            - [distance2mesh.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/distance2mesh.py)

                ![](https://vedo.embl.es/images/basic/distance2mesh.png)
        """
        if bounds is None:
            bounds = self.bounds()
        if max_radius is None:
            max_radius = self.diagonal_size() / 2
        dist = vtki.new("SignedDistance")
        dist.SetInputData(self.dataset)
        dist.SetRadius(max_radius)
        dist.SetBounds(bounds)
        dist.SetDimensions(dims)
        dist.Update()
        img = dist.GetOutput()
        if invert:
            mat = vtki.new("ImageMathematics")
            mat.SetInput1Data(img)
            mat.SetOperationToMultiplyByK()
            mat.SetConstantK(-1)
            mat.Update()
            img = mat.GetOutput()

        vol = vedo.Volume(img)
        vol.name = "SignedDistanceVolume"
        vol.pipeline = utils.OperationNode(
            "signed_distance",
            parents=[self],
            comment=f"dims={tuple(vol.dimensions())}",
            c="#e9c46a:#0096c7",
        )
        return vol

    def unsigned_distance(
        self, dims=(25, 25, 25), bounds=(), max_radius=0, cap_value=0
    ) -> vedo.Volume:
        """
        Compute the `Volume` object whose voxels contains the unsigned distance
        from the input object.
        """
        dist = vtki.new("UnsignedDistance")
        dist.SetInputData(self.dataset)
        dist.SetDimensions(dims)

        if len(bounds) == 6:
            dist.SetBounds(bounds)
        else:
            dist.SetBounds(self.bounds())
        if not max_radius:
            max_radius = self.diagonal_size() / 10
        dist.SetRadius(max_radius)

        if self.point_locator:
            dist.SetLocator(self.point_locator)

        if cap_value is not None:
            dist.CappingOn()
            dist.SetCapValue(cap_value)
        dist.SetOutputScalarTypeToFloat()
        dist.Update()
        vol = vedo.Volume(dist.GetOutput())
        vol.name = "UnsignedDistanceVolume"
        vol.pipeline = utils.OperationNode(
            "unsigned_distance", parents=[self], c="#e9c46a:#0096c7"
        )
        return vol

    def smooth_data(
        self,
        niter=10,
        relaxation_factor=0.1,
        strategy=0,
        mask=None,
        mode="distance2",
        exclude=("Normals", "TextureCoordinates"),
    ) -> Self:
        """
        Smooth point attribute data using distance weighted Laplacian kernel.
        The effect is to blur regions of high variation and emphasize low variation regions.

        A central concept of this method is the point smoothing stencil.
        A smoothing stencil for a point p(i) is the list of points p(j) which connect to p(i) via an edge.
        To smooth the attributes of point p(i), p(i)'s attribute data a(i) are iteratively averaged using
        the distance weighted average of the attributes of a(j) (the weights w[j] sum to 1).
        This averaging process is repeated until the maximum number of iterations is reached.

        The relaxation factor (R) is also important as the smoothing process proceeds in an iterative fashion.
        The a(i+1) attributes are determined from the a(i) attributes as follows:
            a(i+1) = (1-R)*a(i) + R*sum(w(j)*a(j))

        Convergence occurs faster for larger relaxation factors.
        Typically a small number of iterations is required for large relaxation factors,
        and in cases where only points adjacent to the boundary are being smoothed, a single iteration with R=1 may be
        adequate (i.e., just a distance weighted average is computed).

        Warning:
            Certain data attributes cannot be correctly interpolated.
            For example, surface normals are expected to be |n|=1;
            after attribute smoothing this constraint is likely to be violated.
            Other vectors and tensors may suffer from similar issues.
            In such a situation, specify `exclude=...` which will not be smoothed
            (and simply passed through to the output).
            Distance weighting function is based on averaging, 1/r, or 1/r**2 weights, where r is the distance
            between the point to be smoothed and an edge connected neighbor (defined by the smoothing stencil).
            The weights are normalized so that sum(w(i))==1. When smoothing based on averaging,
            the weights are simply 1/n, where n is the number of connected points in the stencil.
            The smoothing process reduces high frequency information in the data attributes.
            With excessive smoothing (large numbers of iterations, and/or a large relaxation factor)
            important details may be lost, and the attributes will move towards an "average" value.
            While this filter will process any dataset type, if the input data is a 3D image volume,
            it's likely much faster to use an image-based algorithm to perform data smoothing.
            To determine boundary points in polygonal data, edges used by only one cell are considered boundary
            (and hence the associated points defining the edge).

        Args:
            niter (int):
                number of iterations
            relaxation_factor (float):
                relaxation factor controlling the amount of Laplacian smoothing applied
            strategy (int):
                strategy to use for Laplacian smoothing

                    - 0: use all points, all point data attributes are smoothed
                    - 1: smooth all point attribute data except those on the boundary
                    - 2: only point data connected to a boundary point are smoothed

            mask (str, np.ndarray):
                array to be used as a mask (ignore then the strategy keyword)
            mode (str):
                smoothing mode, either "distance2", "distance" or "average"

                    - distance**2 weighted (i.e., 1/r**2 interpolation weights)
                    - distance weighted (i.e., 1/r) approach;
                    - simple average of all connected points in the stencil

            exclude (list):
                list of arrays to be excluded from smoothing

        See also: `laplacian_diffusion()`, [diffuse_data.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/diffuse_data.py)
        """
        saf = vtki.new("AttributeSmoothingFilter")
        saf.SetInputData(self.dataset)
        saf.SetRelaxationFactor(relaxation_factor)
        saf.SetNumberOfIterations(niter)

        for ex in exclude:
            saf.AddExcludedArray(ex)

        if mode == "distance":
            saf.SetWeightsTypeToDistance()
        elif mode == "distance2":
            saf.SetWeightsTypeToDistance2()
        elif mode == "average":
            saf.SetWeightsTypeToAverage()
        else:
            vedo.logger.error(f"smooth_data(): unknown mode '{mode}'")
            raise TypeError(f"smooth_data(): unknown mode '{mode}', use 'distance2', 'distance', or 'average'")

        if mask is not None:
            saf.SetSmoothingStrategyToSmoothingMask()
            if isinstance(mask, str):
                mask_ = self.dataset.GetPointData().GetArray(mask)
                if not mask_:
                    vedo.logger.error(f"smooth_data(): mask array {mask} not found")
                    return self
                mask_array = vtki.vtkUnsignedCharArray()
                mask_array.ShallowCopy(mask_)
                mask_array.SetName(mask_.GetName())
            else:
                mask_array = utils.numpy2vtk(mask, dtype=np.uint8)
            saf.SetSmoothingMask(mask_array)
        else:
            saf.SetSmoothingStrategy(strategy)

        saf.Update()

        self._update(saf.GetOutput())
        self.pipeline = utils.OperationNode(
            "smooth_data", comment=f"strategy {strategy}", parents=[self], c="#9e2a2b"
        )
        return self

    def compute_streamlines(
        self,
        seeds: Any,
        integrator="rk4",
        direction="forward",
        initial_step_size=None,
        max_propagation=None,
        max_steps=10000,
        step_length=0,
        surface_constrained=False,
        compute_vorticity=False,
    ) -> vedo.Lines | None:
        """
        Integrate a vector field to generate streamlines.

        Args:
            seeds (Mesh, Points, list):
                starting points of the streamlines
            integrator (str):
                type of integration method to be used:

                    - rk2: Runge-Kutta 2
                    - rk4: Runge-Kutta 4
                    - rk45: Runge-Kutta 45

            direction (str):
                direction of integration, either "forward", "backward" or "both"
            initial_step_size (float):
                initial step size used for line integration
            max_propagation (float):
                maximum length of a streamline expressed in absolute units
            max_steps (int):
                maximum number of steps for a streamline
            step_length (float):
                maximum length of a step expressed in absolute units
            surface_constrained (bool):
                whether to stop integrating when the streamline leaves the surface
            compute_vorticity (bool):
                whether to compute the vorticity at each streamline point
        """
        b = self.dataset.GetBounds()
        size = (b[5] - b[4] + b[3] - b[2] + b[1] - b[0]) / 3
        if initial_step_size is None:
            initial_step_size = size / 1000.0

        if max_propagation is None:
            max_propagation = size * 2

        if utils.is_sequence(seeds):
            seeds = vedo.Points(seeds)

        sti = vtki.new("StreamTracer")
        sti.SetSourceData(seeds.dataset)
        if isinstance(self, vedo.RectilinearGrid):
            sti.SetInputData(vedo.UnstructuredGrid(self.dataset).dataset)
        else:
            sti.SetInputDataObject(self.dataset)

        sti.SetInitialIntegrationStep(initial_step_size)
        sti.SetComputeVorticity(compute_vorticity)
        sti.SetMaximumNumberOfSteps(max_steps)
        sti.SetMaximumPropagation(max_propagation)
        sti.SetSurfaceStreamlines(surface_constrained)
        if step_length:
            sti.SetMaximumIntegrationStep(step_length)

        if "for" in direction:
            sti.SetIntegrationDirectionToForward()
        elif "back" in direction:
            sti.SetIntegrationDirectionToBackward()
        elif "both" in direction:
            sti.SetIntegrationDirectionToBoth()
        else:
            vedo.logger.error(
                f"in compute_streamlines(), unknown direction {direction}"
            )
            return None

        if integrator == "rk2":
            sti.SetIntegratorTypeToRungeKutta2()
        elif integrator == "rk4":
            sti.SetIntegratorTypeToRungeKutta4()
        elif integrator == "rk45":
            sti.SetIntegratorTypeToRungeKutta45()
        else:
            vedo.logger.error(
                f"in compute_streamlines(), unknown integrator {integrator}"
            )
            return None

        sti.Update()

        stlines = vedo.shapes.Lines(sti.GetOutput(), lw=4)
        stlines.name = "StreamLines"
        self.pipeline = utils.OperationNode(
            "compute_streamlines",
            comment=f"{integrator}",
            parents=[self, seeds],
            c="#9e2a2b",
        )
        return stlines


###############################################################################
