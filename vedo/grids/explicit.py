#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import time
from weakref import ref as weak_ref_to
from typing import Any
from typing_extensions import Self
import numpy as np

import vedo.vtkclasses as vtki  # a wrapper for lazy imports

import vedo
from vedo import utils
from vedo.core import PointAlgorithms
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

    # int 	GetDataDimension ()
    #  	Return the dimensionality of the data.
    
    # void 	GetCellDims (int cellDims[3])
    #  	Computes the cell dimensions according to internal point dimensions.
    
    # int 	GetExtentType () override
    #  	The extent type is a 3D extent.
    
    # void 	BuildLinks ()
    #  	Build topological links from points to lists of cells that use each point.
    
    # vtkIdType * 	GetCellPoints (vtkIdType cellId)
    #  	Get direct raw pointer to the 8 points indices of an hexahedra.
    
    # void 	GetCellPoints (vtkIdType cellId, vtkIdType &npts, vtkIdType *&pts)
    #  	More efficient method to obtain cell points.
    
    # void 	GetCellPoints (vtkIdType cellId, vtkIdType &npts, vtkIdType const *&pts, vtkIdList *ptIds) override
    #  	More efficient method to obtain cell points.
    
    # void 	GetCellNeighbors (vtkIdType cellId, vtkIdType neighbors[6], int *wholeExtent=nullptr)
    #  	Get cell neighbors of the cell for every faces.
    
    # void 	ComputeCellStructuredCoords (vtkIdType cellId, int &i, int &j, int &k, bool adjustForExtent=true)
    #  	Given a cellId, get the structured coordinates (i-j-k).
    
    # vtkIdType 	ComputeCellId (int i, int j, int k, bool adjustForExtent=true)
    #  	Given a location in structured coordinates (i-j-k), return the cell id.
    
    # void 	ComputeFacesConnectivityFlagsArray ()
    #  	Compute the faces connectivity flags array.
    
    # bool 	HasAnyBlankCells () override
    #  	Returns true if one or more cells are blanked, false otherwise.
    
    # unsigned char 	IsCellVisible (vtkIdType cellId)
    #  	Return non-zero value if specified cell is visible.
    
    # unsigned char 	IsCellGhost (vtkIdType cellId)
    #  	Return non-zero value if specified cell is a ghost cell.
    
    # bool 	HasAnyGhostCells ()
    #  	Returns true if one or more cells are ghost, false otherwise.
    
    # void 	CheckAndReorderFaces ()
    #  	Check faces are numbered correctly regarding ijk numbering 
    # If not this will reorganize cell points order so face order is valid.
    
    # void 	GetCellBounds (vtkIdType cellId, double bounds[6]) override
    #  	Standard vtkDataSet API methods.
    
    # int 	GetCellType (vtkIdType cellId) override
    #  	Standard vtkDataSet API methods.
    
    # vtkIdType 	GetCellSize (vtkIdType cellId) override
    #  	Standard vtkDataSet API methods.
    
    # vtkIdType 	GetNumberOfCells () override
    #  	Standard vtkDataSet API methods.
    
    # void 	GetCellPoints (vtkIdType cellId, vtkIdList *ptIds) override
    #  	Standard vtkDataSet API methods.
    
    # void 	GetPointCells (vtkIdType ptId, vtkIdList *cellIds) override
    #  	Standard vtkDataSet API methods.
    
    # int 	GetMaxCellSize () override
    #  	Standard vtkDataSet API methods.
    
    # int 	GetMaxSpatialDimension () override
    #  	Standard vtkDataSet API methods.
    
    # int 	GetMinSpatialDimension () override
    #  	Standard vtkDataSet API methods.
    
    # void 	GetCellNeighbors (vtkIdType cellId, vtkIdList *ptIds, vtkIdList *cellIds) override
    #  	Standard vtkDataSet API methods.
    
    # void 	SetDimensions (int i, int j, int k)
    #  	Set/Get the dimensions of this structured dataset in term of number of points along each direction.
    
    # void 	SetDimensions (int dim[3])
    #  	Set/Get the dimensions of this structured dataset in term of number of points along each direction.
    
    # void 	GetDimensions (int dim[3])
    #  	Set/Get the dimensions of this structured dataset in term of number of points along each direction.
    
    # void 	SetExtent (int x0, int x1, int y0, int y1, int z0, int z1)
    #  	Set/Get the extent of this structured dataset in term of number of points along each direction.
    
    # void 	SetExtent (int extent[6])
    #  	Set/Get the extent of this structured dataset in term of number of points along each direction.

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
            self.dataset = vtki.vtkExplicitStructuredGrid()

        elif isinstance(inputobj, vtki.vtkExplicitStructuredGrid):
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
            self.dataset = vtki.vtkExplicitStructuredGrid()
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
    
    def dimensions(self) -> np.ndarray:
        """Return the number of points in the x, y and z directions."""
        return np.array(self.dataset.GetDimensions())
    
    def clone(self, deep=True) -> ExplicitStructuredGrid:
        """Return a clone copy of the StructuredGrid. Alias of `copy()`."""
        if deep:
            newrg = vtki.vtkExplicitStructuredGrid()
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


