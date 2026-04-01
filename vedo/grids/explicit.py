#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import time
from weakref import ref as weak_ref_to
from typing import Any
from typing_extensions import Self
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkExplicitStructuredGrid as vtkExplicitStructuredGrid_

import vedo.vtkclasses as vtki  # a wrapper for lazy imports

import vedo
from vedo import utils
from vedo.core import PointAlgorithms
from vedo.core.summary import (
    active_array_label,
    format_bounds,
    summarize_array,
    summary_panel,
    summary_string,
)
from vedo.mesh import Mesh
from vedo.file_io import download
from vedo.visual import MeshVisual
from vedo.core.transformations import LinearTransform
from .unstructured import UnstructuredGrid

class ExplicitStructuredGrid:
    """
    Build an explicit structured grid.

    An explicit structured grid is a dataset where edges of the hexahedrons are
    not necessarily parallel to the coordinate axes.
    It can be thought of as a tessellation of a block of 3D space,
    similar to a `RectilinearGrid`
    except that the cells are not necessarily cubes, they can have different
    orientations but are connected in the same way as a `RectilinearGrid`.

    Arguments:
        inputobj : (vtkExplicitStructuredGrid, list, str)
            list of points and indices, or filename
    """

    def __init__(self, inputobj=None):
        """
        A StructuredGrid is a dataset where edges of the hexahedrons are
        not necessarily parallel to the coordinate axes.
        It can be thought of as a tessellation of a block of 3D space,
        similar to a `RectilinearGrid`
        except that the cells are not necessarily cubes, they can have different
        orientations but are connected in the same way as a `RectilinearGrid`.

        Arguments:
            inputobj : (vtkExplicitStructuredGrid, list, str)
                list of points and indices, or filename"
                """
        self.dataset = None
        self.mapper = vtki.new("PolyDataMapper")
        self._actor = vtki.vtkActor()
        self._actor.retrieve_object = weak_ref_to(self)
        self._actor.SetMapper(self.mapper)
        self.properties = self._actor.GetProperty()

        self.transform = LinearTransform()
        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None

        self.name = "ExplicitStructuredGrid"
        self.filename = ""

        self.info = {}
        self.time =  time.time()

        ###############################
        if inputobj is None:
            self.dataset = vtkExplicitStructuredGrid_()

        elif isinstance(inputobj, vtkExplicitStructuredGrid_):
            self.dataset = inputobj

        elif isinstance(inputobj, ExplicitStructuredGrid):
            self.dataset = inputobj.dataset

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            if inputobj.endswith(".vts"):
                reader = vtki.new("XMLExplicitStructuredGridReader")
            else:
                reader = vtki.new("ExplicitStructuredGridReader")
            self.filename = inputobj
            reader.SetFileName(inputobj)
            reader.Update()
            self.dataset = reader.GetOutput()

        elif utils.is_sequence(inputobj):
            self.dataset = vtkExplicitStructuredGrid_()
            x, y, z = inputobj
            xyz = np.vstack((
                x.flatten(order="F"),
                y.flatten(order="F"),
                z.flatten(order="F"))
            ).T
            dims = x.shape
            self.dataset.SetDimensions(dims)
            # self.dataset.SetDimensions(dims[1], dims[0], dims[2])
            vpoints = vtki.vtkPoints()
            vpoints.SetData(utils.numpy2vtk(xyz))
            self.dataset.SetPoints(vpoints)


        ###############################
        if not self.dataset:
            vedo.logger.error(f"ExplicitStructuredGrid: cannot understand input type {type(inputobj)}")
            return

        self.properties.SetColor(0.352, 0.612, 0.996)  # blue7
        self.pipeline = utils.OperationNode(
            self, comment=f"#cells {self.dataset.GetNumberOfCells()}", c="#9e2a2b"
        )

    @property
    def actor(self):
        """Return the `vtkActor` of the object."""
        gf = vtki.new("GeometryFilter")
        gf.SetInputData(self.dataset)
        gf.Update()
        self.mapper.SetInputData(gf.GetOutput())
        self.mapper.Modified()
        return self._actor
    
    @actor.setter
    def actor(self, _):
        pass

    def _update(self, data, reset_locators=False):
        self.dataset = data
        if reset_locators:
            self.cell_locator = None
            self.point_locator = None
        return self

    def __str__(self):
        return summary_string(self, self._summary_rows(), color="cyan")

    def __repr__(self):
        return self.__str__()

    def __rich__(self):
        return summary_panel(self, self._summary_rows(), color="cyan")

    def print(self):
        """Print object info."""
        print(self)
        return self

    def _summary_rows(self):
        rows = [("name", str(self.name))]
        if self.filename:
            rows.append(("filename", str(self.filename)))
        rows.append(("dimensions", str(self.dimensions())))
        rows.append(("cell dimensions", str(self.cell_dimensions())))
        rows.append(("data dimension", str(self.data_dimension())))
        rows.append(("center", utils.precision(self.dataset.GetCenter(), 6)))
        rows.append(("bounds", format_bounds(self.dataset.GetBounds(), utils.precision)))
        rows.append(("memory size", utils.precision(self.dataset.GetActualMemorySize() / 1024, 2) + " MB"))
        rows.append(("blank points", str(self.has_blank_points())))
        rows.append(("blank cells", str(self.has_blank_cells())))
        rows.append(("ghost points", str(self.has_ghost_points())))
        rows.append(("ghost cells", str(self.has_ghost_cells())))

        point_data = self.dataset.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            key = point_data.GetArrayName(i)
            if not key:
                continue
            arr = point_data.GetArray(key)
            if arr is None:
                continue
            narr = utils.vtk2numpy(arr)
            label = active_array_label(self.dataset, "point", key, "pointdata")
            rows.append((label, f'"{key}" ' + summarize_array(narr, utils.precision, dim_label="ndim")))

        cell_data = self.dataset.GetCellData()
        for i in range(cell_data.GetNumberOfArrays()):
            key = cell_data.GetArrayName(i)
            if not key:
                continue
            arr = cell_data.GetArray(key)
            if arr is None:
                continue
            narr = utils.vtk2numpy(arr)
            label = active_array_label(self.dataset, "cell", key, "celldata")
            rows.append((label, f'"{key}" ' + summarize_array(narr, utils.precision, dim_label="ndim")))

        field_data = self.dataset.GetFieldData()
        for i in range(field_data.GetNumberOfArrays()):
            arr = field_data.GetAbstractArray(i)
            if arr is None or not arr.GetName():
                continue
            rows.append(("metadata", f'"{arr.GetName()}" ({arr.GetNumberOfTuples()} values)'))
        return rows
    
    def dimensions(self) -> np.ndarray:
        """Return the number of points in the x, y and z directions."""
        try:
            dims = self.dataset.GetDimensions()
        except TypeError:
            dims = [0, 0, 0]
            self.dataset.GetDimensions(dims)
        return np.array(dims)

    def data_dimension(self) -> int:
        """Return the dimensionality of the data."""
        return self.dataset.GetDataDimension()

    def cell_dimensions(self) -> np.ndarray:
        """Return the number of cells in the x, y and z directions."""
        dims = [0, 0, 0]
        self.dataset.GetCellDims(dims)
        return np.array(dims)

    def extent(self) -> np.ndarray:
        """Return the structured grid extent."""
        return np.array(self.dataset.GetExtent())

    def extent_type(self) -> int:
        """Return the extent type identifier."""
        return self.dataset.GetExtentType()

    def set_dimensions(self, *dims) -> Self:
        """Set the grid dimensions as number of points along x, y and z."""
        if len(dims) == 1 and utils.is_sequence(dims[0]):
            dims = dims[0]
        self.dataset.SetDimensions(*dims)
        self.mapper.Modified()
        return self

    def set_extent(self, *extent) -> Self:
        """Set the structured grid extent."""
        if len(extent) == 1 and utils.is_sequence(extent[0]):
            extent = extent[0]
        self.dataset.SetExtent(*extent)
        self.mapper.Modified()
        return self

    def build_links(self) -> Self:
        """Build topological links from points to the cells that use them."""
        self.dataset.BuildLinks()
        return self

    def cell_points(self, cell_id: int) -> np.ndarray:
        """Return the point ids that define cell `cell_id`."""
        pt_ids = vtki.vtkIdList()
        self.dataset.GetCellPoints(cell_id, pt_ids)
        return np.array([pt_ids.GetId(i) for i in range(pt_ids.GetNumberOfIds())], dtype=int)

    def point_cells(self, point_id: int) -> np.ndarray:
        """Return the cell ids that use point `point_id`."""
        cell_ids = vtki.vtkIdList()
        self.dataset.GetPointCells(point_id, cell_ids)
        return np.array([cell_ids.GetId(i) for i in range(cell_ids.GetNumberOfIds())], dtype=int)

    def cell_neighbors(
        self,
        cell_id: int,
        pt_ids=None,
        whole_extent=None,
    ) -> np.ndarray:
        """
        Return the neighbors of cell `cell_id`.

        If `pt_ids` is given, return the ids of the cells sharing all those points.
        Otherwise return the six face-neighbor ids.
        """
        if pt_ids is None:
            neighbors = [-1] * 6
            if whole_extent is None:
                self.dataset.GetCellNeighbors(cell_id, neighbors)
            else:
                self.dataset.GetCellNeighbors(cell_id, neighbors, whole_extent)
            return np.array(neighbors, dtype=int)

        ids = vtki.vtkIdList()
        for pid in pt_ids:
            ids.InsertNextId(int(pid))
        neighbors = vtki.vtkIdList()
        self.dataset.GetCellNeighbors(cell_id, ids, neighbors)
        return np.array([neighbors.GetId(i) for i in range(neighbors.GetNumberOfIds())], dtype=int)

    def compute_cell_structured_coords(
        self,
        cell_id: int,
        adjust_for_extent=True,
    ) -> np.ndarray:
        """Return the structured `(i, j, k)` coordinates of cell `cell_id`."""
        i = vtki.mutable(0)
        j = vtki.mutable(0)
        k = vtki.mutable(0)
        self.dataset.ComputeCellStructuredCoords(cell_id, i, j, k, adjust_for_extent)
        return np.array([int(i), int(j), int(k)])

    def compute_cellid(self, ijk, adjust_for_extent=True) -> int:
        """Return the cell id for the structured coordinates `(i, j, k)`."""
        if utils.is_sequence(ijk):
            return self.dataset.ComputeCellId(int(ijk[0]), int(ijk[1]), int(ijk[2]), adjust_for_extent)
        raise TypeError("compute_cellid() expects a sequence of 3 structured coordinates")

    def compute_faces_connectivity_flags_array(self) -> Self:
        """Compute the faces connectivity flags array."""
        self.dataset.ComputeFacesConnectivityFlagsArray()
        return self

    def has_blank_points(self) -> bool:
        """Return True if the grid has blank points."""
        return self.dataset.HasAnyBlankPoints()

    def has_blank_cells(self) -> bool:
        """Return True if the grid has blank cells."""
        return self.dataset.HasAnyBlankCells()

    def is_cell_visible(self, cell_id: int) -> bool:
        """Return True if cell `cell_id` is visible."""
        return bool(self.dataset.IsCellVisible(cell_id))

    def is_cell_ghost(self, cell_id: int) -> bool:
        """Return True if cell `cell_id` is marked as ghost."""
        return bool(self.dataset.IsCellGhost(cell_id))

    def has_ghost_cells(self) -> bool:
        """Return True if the grid has ghost cells."""
        return self.dataset.HasAnyGhostCells()

    def has_ghost_points(self) -> bool:
        """Return True if the grid has ghost points."""
        return self.dataset.HasAnyGhostPoints()

    def blank_cell(self, cell_id: int) -> Self:
        """Blank cell `cell_id`."""
        self.dataset.BlankCell(cell_id)
        return self

    def unblank_cell(self, cell_id: int) -> Self:
        """Unblank cell `cell_id`."""
        self.dataset.UnBlankCell(cell_id)
        return self

    def check_and_reorder_faces(self) -> Self:
        """Check and reorder cell faces to match the structured orientation."""
        self.dataset.CheckAndReorderFaces()
        return self

    def cell_bounds(self, cell_id: int) -> np.ndarray:
        """Return the bounds of cell `cell_id`."""
        bounds = [0.0] * 6
        self.dataset.GetCellBounds(cell_id, bounds)
        return np.array(bounds)

    def cell_type(self, cell_id: int) -> int:
        """Return the VTK cell type id of cell `cell_id`."""
        return self.dataset.GetCellType(cell_id)

    def cell_size(self, cell_id: int) -> int:
        """Return the number of points used by cell `cell_id`."""
        return self.dataset.GetCellSize(cell_id)

    def number_of_cells(self) -> int:
        """Return the number of cells."""
        return self.dataset.GetNumberOfCells()

    def max_cell_size(self) -> int:
        """Return the maximum cell size."""
        return self.dataset.GetMaxCellSize()

    def max_spatial_dimension(self) -> int:
        """Return the maximum spatial dimension across all cells."""
        return self.dataset.GetMaxSpatialDimension()

    def min_spatial_dimension(self) -> int:
        """Return the minimum spatial dimension across all cells."""
        return self.dataset.GetMinSpatialDimension()

    def find_point(self, x: list) -> int:
        """Given a position `x`, return the id of the closest point."""
        return self.dataset.FindPoint(x)
    
    def clone(self, deep=True) -> ExplicitStructuredGrid:
        """Return a clone copy of the StructuredGrid. Alias of `copy()`."""
        if deep:
            newrg = vtkExplicitStructuredGrid_()
            newrg.CopyStructure(self.dataset)
            newrg.CopyAttributes(self.dataset)
            newvol = ExplicitStructuredGrid(newrg)
        else:
            newvol = ExplicitStructuredGrid(self.dataset)

        prop = vtki.vtkProperty()
        prop.DeepCopy(self.properties)
        newvol.actor.SetProperty(prop)
        newvol.properties = prop
        newvol.pipeline = utils.OperationNode("clone", parents=[self], c="#bbd0ff", shape="diamond")
        return newvol
    
    def cut_with_plane(self, origin=(0, 0, 0), normal="x") -> vedo.UnstructuredGrid:
        """
        Cut the object with the plane defined by a point and a normal.

        Arguments:
            origin : (list)
                the cutting plane goes through this point
            normal : (list, str)
                normal vector to the cutting plane

        Returns an `UnstructuredGrid` object.
        """
        strn = str(normal)
        if strn   ==  "x": normal = (1, 0, 0)
        elif strn ==  "y": normal = (0, 1, 0)
        elif strn ==  "z": normal = (0, 0, 1)
        elif strn == "-x": normal = (-1, 0, 0)
        elif strn == "-y": normal = (0, -1, 0)
        elif strn == "-z": normal = (0, 0, -1)
        plane = vtki.new("Plane")
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        clipper = vtki.new("ClipDataSet")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(plane)
        clipper.GenerateClipScalarsOff()
        clipper.GenerateClippedOutputOff()
        clipper.SetValue(0)
        clipper.Update()
        cout = clipper.GetOutput()
        ug = vedo.UnstructuredGrid(cout)
        if isinstance(self, vedo.UnstructuredGrid):
            self._update(cout)
            self.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
            return self
        ug.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
        return ug
