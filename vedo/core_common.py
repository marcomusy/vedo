#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common algorithm mixins shared by vedo objects."""

from typing import Union, Any
from typing_extensions import Self
import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils
from vedo.transformations import LinearTransform
from vedo.core_data import DataArrayHelper, _get_data_legacy_format

__all__ = ["CommonAlgorithms"]

class CommonAlgorithms:
    """Common algorithms."""

    def _ensure_cell_locator(cls):
        """Build and cache a cell locator for the current dataset if missing."""
        if not cls.cell_locator:
            cls.cell_locator = vtki.new("CellTreeLocator")
            cls.cell_locator.SetDataSet(cls.dataset)
            cls.cell_locator.BuildLocator()
        return cls.cell_locator

    @staticmethod
    def _vtk_idlist_to_numpy(id_list) -> np.ndarray:
        """Convert a vtkIdList to a numpy array of ids."""
        return np.array([id_list.GetId(i) for i in range(id_list.GetNumberOfIds())])

    def _run_gradient_filter(
        cls,
        mode: str,
        on: str,
        array_name: Union[str, None],
        fast: bool,
        result_name: str,
    ) -> np.ndarray:
        """Shared implementation for gradient/divergence/vorticity extraction."""
        gf = vtki.new("GradientFilter")
        if on.startswith("p"):
            varr = cls.dataset.GetPointData()
            assoc = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS
            getter = lambda out: out.GetPointData().GetArray(result_name)
        elif on.startswith("c"):
            varr = cls.dataset.GetCellData()
            assoc = vtki.vtkDataObject.FIELD_ASSOCIATION_CELLS
            getter = lambda out: out.GetCellData().GetArray(result_name)
        else:
            vedo.logger.error(f"in {mode}(): unknown option {on}")
            raise RuntimeError

        if array_name is None:
            if mode == "gradient":
                active = varr.GetScalars()
                if not active:
                    vedo.logger.error(f"in gradient: no scalars found for {on}")
                    raise RuntimeError
            else:
                active = varr.GetVectors()
                if not active:
                    vedo.logger.error(f"in {mode}(): no vectors found for {on}")
                    raise RuntimeError
            array_name = active.GetName()

        gf.SetInputData(cls.dataset)
        gf.SetInputScalars(assoc, array_name)
        gf.SetFasterApproximation(fast)

        gf.ComputeGradientOn() if mode == "gradient" else gf.ComputeGradientOff()
        gf.ComputeDivergenceOn() if mode == "divergence" else gf.ComputeDivergenceOff()
        gf.ComputeVorticityOn() if mode == "vorticity" else gf.ComputeVorticityOff()

        if mode == "gradient":
            gf.SetResultArrayName(result_name)
        elif mode == "divergence":
            gf.SetDivergenceArrayName(result_name)
        elif mode == "vorticity":
            gf.SetVorticityArrayName(result_name)

        gf.Update()
        return utils.vtk2numpy(getter(gf.GetOutput()))

    @property
    def pointdata(cls):
        """
        Create and/or return a `numpy.array` associated to points (vertices).
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.pointdata["arrayname"]`

        Usage:

            `myobj.pointdata.keys()` to return the available data array names

            `myobj.pointdata.select(name)` to make this array the active one

            `myobj.pointdata.remove(name)` to remove this array
        """
        return DataArrayHelper(cls, 0)

    @property
    def celldata(cls):
        """
        Create and/or return a `numpy.array` associated to cells (faces).
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.celldata["arrayname"]`

        Usage:

            `myobj.celldata.keys()` to return the available data array names

            `myobj.celldata.select(name)` to make this array the active one

            `myobj.celldata.remove(name)` to remove this array
        """
        return DataArrayHelper(cls, 1)

    @property
    def metadata(cls):
        """
        Create and/or return a `numpy.array` associated to neither cells nor faces.
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.metadata["arrayname"]`

        Usage:

            `myobj.metadata.keys()` to return the available data array names

            `myobj.metadata.select(name)` to make this array the active one

            `myobj.metadata.remove(name)` to remove this array
        """
        return DataArrayHelper(cls, 2)

    def rename(cls, newname: str) -> Self:
        """Rename the object"""
        try:
            cls.name = newname
        except AttributeError:
            vedo.logger.error(f"Cannot rename object {cls}")
        return cls

    def memory_address(cls) -> int:
        """
        Return a unique memory address integer which may serve as the ID of the
        object, or passed to c++ code.
        """
        # https://www.linkedin.com/pulse/speedup-your-code-accessing-python-vtk-objects-from-c-pletzer/
        # https://github.com/tfmoraes/polydata_connectivity
        return int(cls.dataset.GetAddressAsString("")[5:], 16)

    def memory_size(cls) -> int:
        """Return the approximate memory size of the object in kilobytes."""
        return cls.dataset.GetActualMemorySize()

    def modified(cls) -> Self:
        """Use in conjunction with `tonumpy()` to update any modifications to the image array."""
        cls.dataset.GetPointData().Modified()
        scals = cls.dataset.GetPointData().GetScalars()
        if scals:
            scals.Modified()
        return cls

    def box(cls, scale=1, padding=0) -> "vedo.Mesh":
        """
        Return the bounding box as a new `Mesh` object.

        Arguments:
            scale : (float)
                box size can be scaled by a factor
            padding : (float, list)
                a constant padding can be added (can be a list `[padx,pady,padz]`)
        """
        b = cls.bounds()
        if not utils.is_sequence(padding):
            padding = [padding, padding, padding]
        length, width, height = b[1] - b[0], b[3] - b[2], b[5] - b[4]
        tol = (length + width + height) / 30000  # useful for boxing text
        pos = [(b[0] + b[1]) / 2, (b[3] + b[2]) / 2, (b[5] + b[4]) / 2 - tol]
        bx = vedo.shapes.Box(
            pos,
            length * scale + padding[0],
            width  * scale + padding[1],
            height * scale + padding[2],
            c="gray",
        )
        try:
            pr = vtki.vtkProperty()
            pr.DeepCopy(cls.properties)
            bx.actor.SetProperty(pr)
            bx.properties = pr
        except (AttributeError, TypeError):
            pass
        bx.flat().lighting("off").wireframe(True)
        return bx

    def update_dataset(cls, dataset, **kwargs) -> Self:
        """Update the dataset of the object with the provided VTK dataset."""
        cls._update(dataset, **kwargs)
        return cls

    def bounds(cls) -> np.ndarray:
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        try:  # this is very slow for large meshes
            pts = cls.vertices
            xmin, ymin, zmin = np.nanmin(pts, axis=0)
            xmax, ymax, zmax = np.nanmax(pts, axis=0)
            return np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        except (AttributeError, ValueError):
            return np.array(cls.dataset.GetBounds())

    def xbounds(cls, i=None) -> np.ndarray:
        """Get the bounds `[xmin,xmax]`. Can specify upper or lower with i (0,1)."""
        b = cls.bounds()
        if i is not None:
            return b[i]
        return np.array([b[0], b[1]])

    def ybounds(cls, i=None) -> np.ndarray:
        """Get the bounds `[ymin,ymax]`. Can specify upper or lower with i (0,1)."""
        b = cls.bounds()
        if i == 0:
            return b[2]
        if i == 1:
            return b[3]
        return np.array([b[2], b[3]])

    def zbounds(cls, i=None) -> np.ndarray:
        """Get the bounds `[zmin,zmax]`. Can specify upper or lower with i (0,1)."""
        b = cls.bounds()
        if i == 0:
            return b[4]
        if i == 1:
            return b[5]
        return np.array([b[4], b[5]])

    def diagonal_size(cls) -> float:
        """Get the length of the diagonal of the bounding box."""
        b = cls.bounds()
        return np.sqrt((b[1] - b[0])**2 + (b[3] - b[2])**2 + (b[5] - b[4])**2)

    def average_size(cls) -> float:
        """
        Calculate and return the average size of the object.
        This is the mean of the vertex distances from the center of mass.
        """
        coords = cls.vertices
        cm = np.mean(coords, axis=0)
        if coords.shape[0] == 0:
            return 0.0
        cc = coords - cm
        return np.mean(np.linalg.norm(cc, axis=1))

    def center_of_mass(cls) -> np.ndarray:
        """Get the center of mass of the object."""
        if isinstance(cls, (vedo.RectilinearGrid, vedo.Volume)):
            return np.array(cls.dataset.GetCenter())
        cmf = vtki.new("CenterOfMass")
        cmf.SetInputData(cls.dataset)
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)

    def copy_data_from(cls, obj: Any) -> Self:
        """Copy all data (point and cell data) from this input object"""
        cls.dataset.GetPointData().PassData(obj.dataset.GetPointData())
        cls.dataset.GetCellData().PassData(obj.dataset.GetCellData())
        cls.pipeline = utils.OperationNode(
            "copy_data_from",
            parents=[cls, obj],
            comment=f"{obj.__class__.__name__}",
            shape="note",
            c="#ccc5b9",
        )
        return cls

    def inputdata(cls):
        """Obsolete, use `.dataset` instead."""
        colors.printc("WARNING: 'inputdata()' is obsolete, use '.dataset' instead.", c="y")
        return cls.dataset

    @property
    def npoints(cls):
        """Retrieve the number of points (or vertices)."""
        return cls.dataset.GetNumberOfPoints()

    @property
    def nvertices(cls):
        """Retrieve the number of vertices (or points)."""
        return cls.dataset.GetNumberOfPoints()

    @property
    def ncells(cls):
        """Retrieve the number of cells."""
        return cls.dataset.GetNumberOfCells()

    def cell_centers(cls, copy_arrays=False) -> "vedo.Points":
        """
        Get the coordinates of the cell centers as a `Points` object.

        Examples:
            - [delaunay2d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/delaunay2d.py)
        """
        vcen = vtki.new("CellCenters")
        vcen.SetCopyArrays(copy_arrays)
        vcen.SetVertexCells(copy_arrays)
        vcen.SetInputData(cls.dataset)
        vcen.Update()
        vpts = vedo.Points(vcen.GetOutput())
        if copy_arrays:
            vpts.copy_properties_from(cls)
        return vpts

    @property
    def lines(cls):
        """
        Get lines connectivity ids as a python array
        formatted as `[[id0,id1], [id3,id4], ...]`

        See also: `lines_as_flat_array()`.
        """
        # Get cell connettivity ids as a 1D array. The vtk format is:
        #    [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        try:
            arr1d = _get_data_legacy_format(cls.dataset.GetLines())
        except AttributeError:
            return np.array([], dtype=int)
        i = 0
        conn = []
        n = len(arr1d)
        for _ in range(n):
            cell = [arr1d[i + k + 1] for k in range(arr1d[i])]
            conn.append(cell)
            i += arr1d[i] + 1
            if i >= n:
                break

        return conn  # cannot always make a numpy array of it!

    @property
    def lines_as_flat_array(cls):
        """
        Get lines connectivity ids as a 1D numpy array.
        Format is e.g. [2,  10,20,  3, 10,11,12,  2, 70,80, ...]

        See also: `lines()`.
        """
        try:
            return _get_data_legacy_format(cls.dataset.GetLines())
        except AttributeError:
            return np.array([], dtype=int)

    def mark_boundaries(cls) -> Self:
        """
        Mark cells and vertices if they lie on a boundary.
        A new array called `BoundaryCells` is added to the object.
        """
        mb = vtki.new("MarkBoundaryFilter")
        mb.SetInputData(cls.dataset)
        mb.Update()
        cls.dataset.DeepCopy(mb.GetOutput())
        cls.pipeline = utils.OperationNode("mark_boundaries", parents=[cls])
        return cls

    def find_cells_in_bounds(cls, xbounds=(), ybounds=(), zbounds=()) -> np.ndarray:
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
            bnds = list(cls.bounds())
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
        cls._ensure_cell_locator().FindCellsWithinBounds(bnds, cell_ids)
        return cls._vtk_idlist_to_numpy(cell_ids)

    def find_cells_along_line(cls, p0, p1, tol=0.001) -> np.ndarray:
        """
        Find cells that are intersected by a line segment.
        """
        cell_ids = vtki.vtkIdList()
        cls._ensure_cell_locator().FindCellsAlongLine(p0, p1, tol, cell_ids)
        return cls._vtk_idlist_to_numpy(cell_ids)

    def find_cells_along_plane(cls, origin, normal, tol=0.001) -> np.ndarray:
        """
        Find cells that are intersected by a plane.
        """
        cell_ids = vtki.vtkIdList()
        cls._ensure_cell_locator().FindCellsAlongPlane(origin, normal, tol, cell_ids)
        return cls._vtk_idlist_to_numpy(cell_ids)

    def keep_cell_types(cls, types=()):
        """
        Extract cells of a specific type.

        Check the VTK cell types here:
        https://vtk.org/doc/nightly/html/vtkCellType_8h.html
        """
        fe = vtki.new("ExtractCellsByType")
        fe.SetInputData(cls.dataset)
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
        cls._update(fe.GetOutput())
        return cls

    def map_cells_to_points(cls, arrays=(), move=False) -> Self:
        """
        Interpolate cell data (i.e., data specified per cell or face)
        into point data (i.e., data specified at each vertex).
        The method of transformation is based on averaging the data values
        of all cells using a particular point.

        A custom list of arrays to be mapped can be passed in input.

        Set `move=True` to delete the original `celldata` array.
        """
        c2p = vtki.new("CellDataToPointData")
        c2p.SetInputData(cls.dataset)
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
        cls._update(c2p.GetOutput(), reset_locators=False)
        cls.mapper.SetScalarModeToUsePointData()
        cls.pipeline = utils.OperationNode("map_cells_to_points", parents=[cls])
        return cls

    @property
    def vertices(cls):
        """
        Return the vertices (points) coordinates.
        This is equivalent to `points` and `coordinates`.
        """
        try:
            # for polydata and unstructured grid
            vpts = cls.dataset.GetPoints()
            if vpts is None:
                return np.array([], dtype=float)
            varr = vpts.GetData()
        except AttributeError:
            # 'vtkImageData' object has no attribute 'GetPoints'
            v2p = vtki.new("ImageToPoints")
            v2p.SetInputData(cls.dataset)
            v2p.Update()
            varr = v2p.GetOutput().GetPoints().GetData()
        except TypeError:
            # for RectilinearGrid, StructuredGrid
            vpts = vtki.vtkPoints()
            cls.dataset.GetPoints(vpts)
            varr = vpts.GetData()
        except Exception as e:
            vedo.logger.error(f"Cannot get point coords for {type(cls)}: {e}")
            return np.array([], dtype=float)

        return utils.vtk2numpy(varr)

    # setter
    @vertices.setter
    def vertices(cls, pts):
        """Set vertex coordinates. Same as `points` and `coordinates`."""
        pts = utils.make3d(pts)
        arr = utils.numpy2vtk(pts, dtype=np.float32)
        try:
            vpts = cls.dataset.GetPoints()
            vpts.SetData(arr)
            vpts.Modified()
        except (AttributeError, TypeError):
            vedo.logger.error(f"Cannot set vertices for {type(cls)}")
            return
        # reset mesh to identity matrix position/rotation:
        cls.point_locator = None
        cls.cell_locator = None
        cls.line_locator = None
        cls.transform = LinearTransform()

    @property
    def points(cls):
        """
        Return the points coordinates. Same as `vertices` and `coordinates`.
        """
        return cls.vertices

    @points.setter
    def points(cls, pts):
        """Set points coordinates. Same as `vertices` and `coordinates`."""
        cls.vertices = pts

    @property
    def coordinates(cls):
        """Return the points coordinates. Same as `vertices` and `points`."""
        return cls.vertices

    @coordinates.setter
    def coordinates(cls, pts):
        """Set points coordinates. Same as `vertices` and `points`."""
        cls.vertices = pts

    @property
    def cells_as_flat_array(cls):
        """
        Get cell connectivity ids as a 1D numpy array.
        Format is e.g. [3,  10,20,30  4, 10,11,12,13  ...]
        """
        try:
            # valid for unstructured grid
            arr1d = _get_data_legacy_format(cls.dataset.GetCells())
        except AttributeError:
            # valid for polydata
            arr1d = _get_data_legacy_format(cls.dataset.GetPolys())
        return arr1d

    @property
    def cells(cls):
        """
        Get the cells connectivity ids as a numpy array.

        The output format is: `[[id0 ... idn], [id0 ... idm],  etc]`.
        """
        try:
            # valid for unstructured grid
            arr1d = _get_data_legacy_format(cls.dataset.GetCells())
        except AttributeError:
            try:
                # valid for polydata
                arr1d = _get_data_legacy_format(cls.dataset.GetPolys())
            except AttributeError:
                vedo.logger.error(f"Cannot get cells for {type(cls)}")
                return np.array([], dtype=int)

        # Get cell connettivity ids as a 1D array. vtk format is:
        # [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        i = 0
        conn = []
        n = len(arr1d)
        if n:
            while True:
                cell = [int(arr1d[i + k]) for k in range(1, arr1d[i] + 1)]
                conn.append(cell)
                i += arr1d[i] + 1
                if i >= n:
                    break
        return conn

    def cell_edge_neighbors(cls):
        """
        Get the cell neighbor indices of each cell.

        Returns a python list of lists.
        """

        def face_to_edges(face):
            edges = []
            size = len(face)
            for i in range(1, size + 1):
                if i == size:
                    edges.append([face[i - 1], face[0]])
                else:
                    edges.append([face[i - 1], face[i]])
            return edges

        pd = cls.dataset
        pd.BuildLinks()

        neicells = []
        for i, cell in enumerate(cls.cells):
            nn = []
            for edge in face_to_edges(cell):
                neighbors = vtki.vtkIdList()
                pd.GetCellEdgeNeighbors(i, edge[0], edge[1], neighbors)
                if neighbors.GetNumberOfIds() > 0:
                    neighbor = neighbors.GetId(0)
                    nn.append(neighbor)
            neicells.append(nn)

        return neicells


    def map_points_to_cells(cls, arrays=(), move=False) -> Self:
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
        p2c.SetInputData(cls.dataset)
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
        cls._update(p2c.GetOutput(), reset_locators=False)
        cls.mapper.SetScalarModeToUseCellData()
        cls.pipeline = utils.OperationNode("map_points_to_cells", parents=[cls])
        return cls

    def resample_data_from(cls, source, tol=None, categorical=False) -> Self:
        """
        Resample point and cell data from another dataset.
        The output has the same structure but its point data have
        the resampled values from target.

        Use `tol` to set the tolerance used to compute whether
        a point in the source is in a cell of the current object.
        Points without resampled values, and their cells, are marked as blank.
        If the data is categorical, then the resulting data will be determined
        by a nearest neighbor interpolation scheme.

        Example:
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
        rs.SetInputData(cls.dataset)
        rs.SetSourceData(source.dataset)

        rs.SetPassPointArrays(True)
        rs.SetPassCellArrays(True)
        rs.SetPassFieldArrays(True)
        rs.SetCategoricalData(categorical)

        rs.SetComputeTolerance(True)
        if tol:
            rs.SetComputeTolerance(False)
            rs.SetTolerance(tol)
        rs.Update()
        cls._update(rs.GetOutput(), reset_locators=False)
        cls.pipeline = utils.OperationNode(
            "resample_data_from",
            comment=f"{source.__class__.__name__}",
            parents=[cls, source],
        )
        return cls

    def interpolate_data_from(
        cls,
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

        Arguments:
            kernel : (str)
                available kernels are [shepard, gaussian, linear]
            null_strategy : (int)
                specify a strategy to use when encountering a "null" point
                during the interpolation process. Null points occur when the local neighborhood
                (of nearby points to interpolate from) is empty.

                - Case 0: an output array is created that marks points
                  as being valid (=1) or null (invalid =0), and the null_value is set as well
                - Case 1: the output data value(s) are set to the provided null_value
                - Case 2: simply use the closest point to perform the interpolation.
            null_value : (float)
                see above.

        Examples:
            - [interpolate_scalar1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolate_scalar1.py)
            - [interpolate_scalar3.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolate_scalar3.py)
            - [interpolate_scalar4.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolate_scalar4.py)
            - [image_probe.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/image_probe.py)

                ![](https://vedo.embl.es/images/advanced/interpolateMeshArray.png)
        """
        if radius is None and not n:
            vedo.logger.error("in interpolate_data_from(): please set either radius or n")
            raise RuntimeError

        if on == "points":
            points = source.dataset
        elif on == "cells":
            c2p = vtki.new("CellDataToPointData")
            c2p.SetInputData(source.dataset)
            c2p.Update()
            points = c2p.GetOutput()
        else:
            vedo.logger.error("in interpolate_data_from(), on must be on points or cells")
            raise RuntimeError()

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
            raise RuntimeError()

        if n:
            kern.SetNumberOfPoints(n)
            kern.SetKernelFootprintToNClosest()
        else:
            kern.SetRadius(radius)
            kern.SetKernelFootprintToRadius()

        # remove arrays that are already present in cls dataset
        # this is because the interpolator will ignore them otherwise
        clsnames = []
        for i in range(cls.dataset.GetPointData().GetNumberOfArrays()):
            name = cls.dataset.GetPointData().GetArrayName(i)
            clsnames.append(name)
        
        pointsnames = []
        for i in range(points.GetPointData().GetNumberOfArrays()):
            name = points.GetPointData().GetArrayName(i)
            pointsnames.append(name)

        for cname in clsnames:
            if cname in set(pointsnames) - set(exclude):
                cls.dataset.GetPointData().RemoveArray(cname)
                # print(f"Removed {cname} from cls dataset")

        interpolator = vtki.new("PointInterpolator")
        interpolator.SetInputData(cls.dataset)
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

        cls._update(cpoly, reset_locators=False)

        cls.pipeline = utils.OperationNode("interpolate_data_from", parents=[cls, source])
        return cls

    def add_ids(cls) -> Self:
        """
        Generate point and cell ids arrays.

        Two new arrays are added to the mesh named `PointID` and `CellID`.
        """
        try:
            ids = vtki.new("IdFilter") # available in VTK <9.6 only
        except AttributeError:
            ids = vtki.new("GenerateIds")
        ids.SetInputData(cls.dataset)
        ids.PointIdsOn()
        ids.CellIdsOn()
        ids.FieldDataOff()
        ids.SetPointIdsArrayName("PointID")
        ids.SetCellIdsArrayName("CellID")
        ids.Update()
        # cls._update(ids.GetOutput(), reset_locators=False) # bug #1267
        point_arr = ids.GetOutput().GetPointData().GetArray("PointID")
        cell_arr  = ids.GetOutput().GetCellData().GetArray("CellID")
        if point_arr:
            cls.dataset.GetPointData().AddArray(point_arr)
        if cell_arr:
            cls.dataset.GetCellData().AddArray(cell_arr)
        cls.pipeline = utils.OperationNode("add_ids", parents=[cls])
        return cls

    def gradient(cls, input_array=None, on="points", fast=False) -> np.ndarray:
        """
        Compute and return the gradiend of the active scalar field as a numpy array.

        Arguments:
            input_array : (str)
                array of the scalars to compute the gradient,
                if None the current active array is selected
            on : (str)
                compute either on 'points' or 'cells' data
            fast : (bool)
                if True, will use a less accurate algorithm
                that performs fewer derivative calculations (and is therefore faster).

        Examples:
            - [isolines.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/isolines.py)

            ![](https://user-images.githubusercontent.com/32848391/72433087-f00a8780-3798-11ea-9778-991f0abeca70.png)
        """
        return cls._run_gradient_filter(
            mode="gradient",
            on=on,
            array_name=input_array,
            fast=fast,
            result_name="Gradient",
        )

    def divergence(cls, array_name=None, on="points", fast=False) -> np.ndarray:
        """
        Compute and return the divergence of a vector field as a numpy array.

        Arguments:
            array_name : (str)
                name of the array of vectors to compute the divergence,
                if None the current active array is selected
            on : (str)
                compute either on 'points' or 'cells' data
            fast : (bool)
                if True, will use a less accurate algorithm
                that performs fewer derivative calculations and is therefore faster.
        """
        return cls._run_gradient_filter(
            mode="divergence",
            on=on,
            array_name=array_name,
            fast=fast,
            result_name="Divergence",
        )

    def vorticity(cls, array_name=None, on="points", fast=False) -> np.ndarray:
        """
        Compute and return the vorticity of a vector field as a numpy array.

        Arguments:
            array_name : (str)
                name of the array to compute the vorticity,
                if None the current active array is selected
            on : (str)
                compute either on 'points' or 'cells' data
            fast : (bool)
                if True, will use a less accurate algorithm
                that performs fewer derivative calculations (and is therefore faster).
        """
        return cls._run_gradient_filter(
            mode="vorticity",
            on=on,
            array_name=array_name,
            fast=fast,
            result_name="Vorticity",
        )

    def probe(
            cls,
            source,
            categorical=False,
            snap=False,
            tol=0,
        ) -> Self:
        """
        Takes a data set and probes its scalars at the specified points in space.

        Note that a mask is also output with valid/invalid points which can be accessed
        with `mesh.pointdata['ValidPointMask']`.

        Arguments:
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
        probe_filter.SetInputData(cls.dataset)
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
        cls._update(probe_filter.GetOutput(), reset_locators=False)
        cls.pipeline = utils.OperationNode("probe", parents=[cls, source])
        cls.pointdata.rename("vtkValidPointMask", "ValidPointMask")
        return cls

    def compute_cell_size(cls) -> Self:
        """
        Add to this object a cell data array
        containing the area, volume and edge length
        of the cells (when applicable to the object type).

        Array names are: `Area`, `Volume`, `Length`.
        """
        csf = vtki.new("CellSizeFilter")
        csf.SetInputData(cls.dataset)
        csf.SetComputeArea(1)
        csf.SetComputeVolume(1)
        csf.SetComputeLength(1)
        csf.SetComputeVertexCount(0)
        csf.SetAreaArrayName("Area")
        csf.SetVolumeArrayName("Volume")
        csf.SetLengthArrayName("Length")
        csf.Update()
        cls._update(csf.GetOutput(), reset_locators=False)
        return cls

    def generate_random_data(cls) -> Self:
        """Fill a dataset with random attributes"""
        gen = vtki.new("RandomAttributeGenerator")
        gen.SetInputData(cls.dataset)
        gen.GenerateAllDataOn()
        gen.SetDataTypeToFloat()
        gen.GeneratePointNormalsOff()
        gen.GeneratePointTensorsOn()
        gen.GenerateCellScalarsOn()
        gen.Update()
        cls._update(gen.GetOutput(), reset_locators=False)
        cls.pipeline = utils.OperationNode("generate_random_data", parents=[cls])
        return cls

    def integrate_data(cls) -> dict:
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
        vinteg.SetInputData(cls.dataset)
        vinteg.Update()
        ugrid = vedo.UnstructuredGrid(vinteg.GetOutput())
        data = dict(
            pointdata=ugrid.pointdata.todict(),
            celldata=ugrid.celldata.todict(),
            metadata=ugrid.metadata.todict(),
        )
        return data

    def write(cls, filename, binary=True) -> None:
        """Write object to file."""
        out = vedo.file_io.write(cls, filename, binary)
        out.pipeline = utils.OperationNode(
            "write", parents=[cls], comment=str(filename)[:15], 
            shape="folder", c="#8a817c"
        )

    def tomesh(cls, bounds=(), shrink=0) -> "vedo.Mesh":
        """
        Extract boundary geometry from dataset (or convert data to polygonal type).

        Two new arrays are added to the mesh: `OriginalCellIds` and `OriginalPointIds`
        to keep track of the original mesh elements.

        Arguments:
            bounds : (list)
                specify a sub-region to extract
            shrink : (float)
                shrink the cells to a fraction of their original size
        """
        geo = vtki.new("GeometryFilter")

        if shrink:
            sf = vtki.new("ShrinkFilter")
            sf.SetInputData(cls.dataset)
            sf.SetShrinkFactor(shrink)
            sf.Update()
            geo.SetInputData(sf.GetOutput())
        else:
            geo.SetInputData(cls.dataset)

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
        msh.pipeline = utils.OperationNode("tomesh", parents=[cls], c="#9e2a2b")
        return msh

    def signed_distance(cls, dims=(20, 20, 20), bounds=None, invert=False, max_radius=None) -> "vedo.Volume":
        """
        Compute the `Volume` object whose voxels contains the signed distance from
        the object. The calling object must have "Normals" defined.

        Arguments:
            bounds : (list, actor)
                bounding box sizes
            dims : (list)
                dimensions (nr. of voxels) of the output volume.
            invert : (bool)
                flip the sign
            max_radius : (float)
                specify how far out to propagate distance calculation

        Examples:
            - [distance2mesh.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/distance2mesh.py)

                ![](https://vedo.embl.es/images/basic/distance2mesh.png)
        """
        if bounds is None:
            bounds = cls.bounds()
        if max_radius is None:
            max_radius = cls.diagonal_size() / 2
        dist = vtki.new("SignedDistance")
        dist.SetInputData(cls.dataset)
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
            parents=[cls],
            comment=f"dims={tuple(vol.dimensions())}",
            c="#e9c46a:#0096c7",
        )
        return vol

    def unsigned_distance(
            cls, dims=(25,25,25), bounds=(), max_radius=0, cap_value=0) -> "vedo.Volume":
        """
        Compute the `Volume` object whose voxels contains the unsigned distance
        from the input object.
        """
        dist = vtki.new("UnsignedDistance")
        dist.SetInputData(cls.dataset)
        dist.SetDimensions(dims)

        if len(bounds) == 6:
            dist.SetBounds(bounds)
        else:
            dist.SetBounds(cls.bounds())
        if not max_radius:
            max_radius = cls.diagonal_size() / 10
        dist.SetRadius(max_radius)

        if cls.point_locator:
            dist.SetLocator(cls.point_locator)

        if cap_value is not None:
            dist.CappingOn()
            dist.SetCapValue(cap_value)
        dist.SetOutputScalarTypeToFloat()
        dist.Update()
        vol = vedo.Volume(dist.GetOutput())
        vol.name = "UnsignedDistanceVolume"
        vol.pipeline = utils.OperationNode(
            "unsigned_distance", parents=[cls], c="#e9c46a:#0096c7")
        return vol

    def smooth_data(cls,
            niter=10, relaxation_factor=0.1, strategy=0, mask=None,
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

        Arguments:
            niter : (int)
                number of iterations
            relaxation_factor : (float)
                relaxation factor controlling the amount of Laplacian smoothing applied
            strategy : (int)
                strategy to use for Laplacian smoothing

                    - 0: use all points, all point data attributes are smoothed
                    - 1: smooth all point attribute data except those on the boundary
                    - 2: only point data connected to a boundary point are smoothed

            mask : (str, np.ndarray)
                array to be used as a mask (ignore then the strategy keyword)
            mode : (str)
                smoothing mode, either "distance2", "distance" or "average"

                    - distance**2 weighted (i.e., 1/r**2 interpolation weights)
                    - distance weighted (i.e., 1/r) approach;
                    - simple average of all connected points in the stencil

            exclude : (list)
                list of arrays to be excluded from smoothing
        """
        saf = vtki.new("AttributeSmoothingFilter")
        saf.SetInputData(cls.dataset)
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
            vedo.logger.error(f"smooth_data(): unknown mode {mode}")
            raise TypeError

        saf.SetSmoothingStrategy(strategy)
        if mask is not None:
            saf.SetSmoothingStrategyToSmoothingMask()
            if isinstance(mask, str):
                mask_ = cls.dataset.GetPointData().GetArray(mask)
                if not mask_:
                    vedo.logger.error(f"smooth_data(): mask array {mask} not found")
                    return cls
                mask_array = vtki.vtkUnsignedCharArray()
                mask_array.ShallowCopy(mask_)
                mask_array.SetName(mask_.GetName())
            else:
                mask_array = utils.numpy2vtk(mask, dtype=np.uint8)
            saf.SetSmoothingMask(mask_array)

        saf.Update()

        cls._update(saf.GetOutput())
        cls.pipeline = utils.OperationNode(
            "smooth_data", comment=f"strategy {strategy}", parents=[cls], c="#9e2a2b"
        )
        return cls

    def compute_streamlines(
            cls,
            seeds: Any,
            integrator="rk4",
            direction="forward",
            initial_step_size=None,
            max_propagation=None,
            max_steps=10000,
            step_length=0,
            surface_constrained=False,
            compute_vorticity=False,
        ) -> Union["vedo.Lines", None]:
        """
        Integrate a vector field to generate streamlines.

        Arguments:
            seeds : (Mesh, Points, list)
                starting points of the streamlines
            integrator : (str)
                type of integration method to be used:

                    - "rk2" (Runge-Kutta 2)
                    - "rk4" (Runge-Kutta 4)
                    - "rk45" (Runge-Kutta 45)

            direction : (str)
                direction of integration, either "forward", "backward" or "both"
            initial_step_size : (float)
                initial step size used for line integration
            max_propagation : (float)
                maximum length of a streamline expressed in absolute units
            max_steps : (int)
                maximum number of steps for a streamline
            step_length : (float)
                maximum length of a step expressed in absolute units
            surface_constrained : (bool)
                whether to stop integrating when the streamline leaves the surface
            compute_vorticity : (bool)
                whether to compute the vorticity at each streamline point
        """
        b = cls.dataset.GetBounds()
        size = (b[5]-b[4] + b[3]-b[2] + b[1]-b[0]) / 3
        if initial_step_size is None:
            initial_step_size = size / 1000.0

        if max_propagation is None:
            max_propagation = size * 2

        if utils.is_sequence(seeds):
            seeds = vedo.Points(seeds)

        sti = vtki.new("StreamTracer")
        sti.SetSourceData(seeds.dataset)
        if isinstance(cls, vedo.RectilinearGrid):
            sti.SetInputData(vedo.UnstructuredGrid(cls.dataset).dataset)
        else:
            sti.SetInputDataObject(cls.dataset)

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
            vedo.logger.error(f"in compute_streamlines(), unknown direction {direction}")
            return None

        if integrator == "rk2":
            sti.SetIntegratorTypeToRungeKutta2()
        elif integrator == "rk4":
            sti.SetIntegratorTypeToRungeKutta4()
        elif integrator == "rk45":
            sti.SetIntegratorTypeToRungeKutta45()
        else:
            vedo.logger.error(f"in compute_streamlines(), unknown integrator {integrator}")
            return None

        sti.Update()

        stlines = vedo.shapes.Lines(sti.GetOutput(), lw=4)
        stlines.name = "StreamLines"
        cls.pipeline = utils.OperationNode(
            "compute_streamlines", comment=f"{integrator}", parents=[cls, seeds], c="#9e2a2b"
        )
        return stlines

###############################################################################
