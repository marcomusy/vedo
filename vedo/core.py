#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Union, Any
from typing_extensions import Self

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils
from vedo.transformations import LinearTransform, NonLinearTransform


__docformat__ = "google"

__doc__ = """
Base classes providing functionality to different vedo objects.

![](https://vedo.embl.es/images/feats/algorithms_illustration.png)
"""

__all__ = [
    "DataArrayHelper",
    "CommonAlgorithms",
    "PointAlgorithms",
    "VolumeAlgorithms",
]


###############################################################################
# warnings = dict(
#     func_name=(
#         "WARNING: some message"
#         " (silence this with vedo.core.warnings['func_name']=False)"
#     ),
# )
# ### USAGE
# def func_name(self):
#     """Obsolete, use ... instead."""
#         if warnings["func_name"]:
#             colors.printc(warnings["func_name"], c="y")
#             warnings["func_name"] = ""
#         return


###############################################################################
class DataArrayHelper:
    """
    Helper class to manage data associated to either points (or vertices) and cells (or faces).

    Internal use only.
    """
    def __init__(self, obj, association):

        self.obj = obj
        self.association = association

    def __getitem__(self, key):

        if self.association == 0:
            data = self.obj.dataset.GetPointData()

        elif self.association == 1:
            data = self.obj.dataset.GetCellData()

        elif self.association == 2:
            data = self.obj.dataset.GetFieldData()

            varr = data.GetAbstractArray(key)
            if isinstance(varr, vtki.vtkStringArray):
                if isinstance(key, int):
                    key = data.GetArrayName(key)
                n = varr.GetNumberOfValues()
                narr = [varr.GetValue(i) for i in range(n)]
                return narr
                ###########

        else:
            raise RuntimeError()

        if isinstance(key, int):
            key = data.GetArrayName(key)

        arr = data.GetArray(key)
        if not arr:
            return None
        return utils.vtk2numpy(arr)

    def __setitem__(self, key, input_array):

        if self.association == 0:
            data = self.obj.dataset.GetPointData()
            n = self.obj.dataset.GetNumberOfPoints()
            self.obj.mapper.SetScalarModeToUsePointData()

        elif self.association == 1:
            data = self.obj.dataset.GetCellData()
            n = self.obj.dataset.GetNumberOfCells()
            self.obj.mapper.SetScalarModeToUseCellData()

        elif self.association == 2:
            data = self.obj.dataset.GetFieldData()
            if not utils.is_sequence(input_array):
                input_array = [input_array]

            if isinstance(input_array[0], str):
                varr = vtki.vtkStringArray()
                varr.SetName(key)
                varr.SetNumberOfComponents(1)
                varr.SetNumberOfTuples(len(input_array))
                for i, iarr in enumerate(input_array):
                    if isinstance(iarr, np.ndarray):
                        iarr = iarr.tolist()  # better format
                        # Note: a string k can be converted to numpy with
                        # import json; k = np.array(json.loads(k))
                    varr.InsertValue(i, str(iarr))
            else:
                try:
                    varr = utils.numpy2vtk(input_array, name=key)
                except TypeError as e:
                    vedo.logger.error(
                        f"cannot create metadata with input object:\n"
                        f"{input_array}"
                        f"\n\nAllowed content examples are:\n"
                        f"- flat list of strings ['a','b', 1, [1,2,3], ...]"
                        f" (first item must be a string in this case)\n"
                        f"  hint: use k = np.array(json.loads(k)) to convert strings\n"
                        f"- numpy arrays of any shape"
                    )
                    raise e

            data.AddArray(varr)
            return  ############

        else:
            raise RuntimeError()

        if len(input_array) != n:
            vedo.logger.error(
                f"Error in point/cell data: length of input {len(input_array)}"
                f" !=  {n} nr. of elements"
            )
            raise RuntimeError()

        input_array = np.asarray(input_array)
        varr = utils.numpy2vtk(input_array, name=key)
        data.AddArray(varr)

        if len(input_array.shape) == 1:  # scalars
            data.SetActiveScalars(key)
            try:  # could be a volume mapper
                self.obj.mapper.SetScalarRange(data.GetScalars().GetRange())
            except AttributeError:
                pass
        elif len(input_array.shape) == 2 and input_array.shape[1] == 3:  # vectors
            if key.lower() == "normals":
                data.SetActiveNormals(key)
            else:
                data.SetActiveVectors(key)

    def keys(self) -> List[str]:
        """Return the list of available data array names"""
        if self.association == 0:
            data = self.obj.dataset.GetPointData()
        elif self.association == 1:
            data = self.obj.dataset.GetCellData()
        elif self.association == 2:
            data = self.obj.dataset.GetFieldData()
        arrnames = []
        for i in range(data.GetNumberOfArrays()):
            name = ""
            if self.association == 2:
                name = data.GetAbstractArray(i).GetName()
            else:
                iarr = data.GetArray(i)
                if iarr:
                    name = iarr.GetName()
            if name:
                arrnames.append(name)
        return arrnames

    def items(self) -> List:
        """Return the list of available data array `(names, values)`."""
        if self.association == 0:
            data = self.obj.dataset.GetPointData()
        elif self.association == 1:
            data = self.obj.dataset.GetCellData()
        elif self.association == 2:
            data = self.obj.dataset.GetFieldData()
        arrnames = []
        for i in range(data.GetNumberOfArrays()):
            if self.association == 2:
                name = data.GetAbstractArray(i).GetName()
            else:
                name = data.GetArray(i).GetName()
            if name:
                arrnames.append((name, self[name]))
        return arrnames

    def todict(self) -> dict:
        """Return a dictionary of the available data arrays."""
        return dict(self.items())

    def rename(self, oldname: str, newname: str) -> None:
        """Rename an array"""
        if self.association == 0:
            varr = self.obj.dataset.GetPointData().GetArray(oldname)
        elif self.association == 1:
            varr = self.obj.dataset.GetCellData().GetArray(oldname)
        elif self.association == 2:
            varr = self.obj.dataset.GetFieldData().GetAbstractArray(oldname)
        if varr:
            varr.SetName(newname)
        else:
            vedo.logger.warning(
                f"Cannot rename non existing array {oldname} to {newname}"
            )

    def remove(self, key: Union[int, str]) -> None:
        """Remove a data array by name or number"""
        if self.association == 0:
            self.obj.dataset.GetPointData().RemoveArray(key)
        elif self.association == 1:
            self.obj.dataset.GetCellData().RemoveArray(key)
        elif self.association == 2:
            self.obj.dataset.GetFieldData().RemoveArray(key)

    def clear(self) -> None:
        """Remove all data associated to this object"""
        if self.association == 0:
            data = self.obj.dataset.GetPointData()
        elif self.association == 1:
            data = self.obj.dataset.GetCellData()
        elif self.association == 2:
            data = self.obj.dataset.GetFieldData()
        for i in range(data.GetNumberOfArrays()):
            if self.association == 2:
                if data.GetAbstractArray(i):
                    name = data.GetAbstractArray(i).GetName()
                    data.RemoveArray(name)
            else:
                if data.GetArray(i):
                    name = data.GetArray(i).GetName()
                    data.RemoveArray(name)

    def select(self, key: Union[int, str]) -> Any:
        """Select one specific array by its name to make it the `active` one."""
        # Default (ColorModeToDefault): unsigned char scalars are treated as colors,
        # and NOT mapped through the lookup table, while everything else is.
        # ColorModeToDirectScalar extends ColorModeToDefault such that all integer
        # types are treated as colors with values in the range 0-255
        # and floating types are treated as colors with values in the range 0.0-1.0.
        # Setting ColorModeToMapScalars means that all scalar data will be mapped
        # through the lookup table.
        # (Note that for multi-component scalars, the particular component
        # to use for mapping can be specified using the SelectColorArray() method.)
        if self.association == 0:
            data = self.obj.dataset.GetPointData()
            self.obj.mapper.SetScalarModeToUsePointData()
        else:
            data = self.obj.dataset.GetCellData()
            self.obj.mapper.SetScalarModeToUseCellData()

        if isinstance(key, int):
            key = data.GetArrayName(key)

        arr = data.GetArray(key)
        if not arr:
            return self.obj

        nc = arr.GetNumberOfComponents()
        # print("GetNumberOfComponents", nc)
        if nc == 1:
            data.SetActiveScalars(key)
        elif nc == 2:
            data.SetTCoords(arr)
        elif nc in (3, 4):
            if "rgb" in key.lower(): # type: ignore
                data.SetActiveScalars(key)
                try:
                    # could be a volume mapper
                    self.obj.mapper.SetColorModeToDirectScalars()
                    data.SetActiveVectors(None) # need this to fix bug in #1066
                    # print("SetColorModeToDirectScalars for", key)
                except AttributeError:
                    pass
            else:
                data.SetActiveVectors(key)
        elif nc == 9:
            data.SetActiveTensors(key)
        else:
            vedo.logger.error(f"Cannot select array {key} with {nc} components")
            return self.obj

        try:
            # could be a volume mapper
            self.obj.mapper.SetArrayName(key)
            self.obj.mapper.ScalarVisibilityOn()
        except AttributeError:
            pass

        return self.obj

    def select_texture_coords(self, key: Union[int,str]) -> Any:
        """Select one specific array to be used as texture coordinates."""
        if self.association == 0:
            data = self.obj.dataset.GetPointData()
            if isinstance(key, int):
                key = data.GetArrayName(key)
            data.SetTCoords(data.GetArray(key))
        else:
            vedo.logger.warning("texture coordinates are only available for point data")
        return self.obj

    def select_normals(self, key: Union[int,str]) -> Any:
        """Select one specific normal array by its name to make it the "active" one."""
        if self.association == 0:
            data = self.obj.dataset.GetPointData()
            self.obj.mapper.SetScalarModeToUsePointData()
        else:
            data = self.obj.dataset.GetCellData()
            self.obj.mapper.SetScalarModeToUseCellData()

        if isinstance(key, int):
            key = data.GetArrayName(key)
        data.SetActiveNormals(key)
        return self.obj

    def print(self, **kwargs) -> None:
        """Print the array names available to terminal"""
        colors.printc(self.keys(), **kwargs)

    def __repr__(self) -> str:
        """Representation"""

        def _get_str(pd, header):
            out = f"\x1b[2m\x1b[1m\x1b[7m{header}"
            if pd.GetNumberOfArrays():
                if self.obj.name:
                    out += f" in {self.obj.name}"
                out += f" contains {pd.GetNumberOfArrays()} array(s)\x1b[0m"
                for i in range(pd.GetNumberOfArrays()):
                    varr = pd.GetArray(i)
                    out += f"\n\x1b[1m\x1b[4mArray name    : {varr.GetName()}\x1b[0m"
                    out += "\nindex".ljust(15) + f": {i}"
                    t = varr.GetDataType()
                    if t in vtki.array_types:
                        out += "\ntype".ljust(15)
                        out += f": {vtki.array_types[t]}"
                    shape = (varr.GetNumberOfTuples(), varr.GetNumberOfComponents())
                    out += "\nshape".ljust(15) + f": {shape}"
                    out += "\nrange".ljust(15) + f": {np.array(varr.GetRange())}"
                    out += "\nmax id".ljust(15) + f": {varr.GetMaxId()}"
                    out += "\nlook up table".ljust(15) + f": {bool(varr.GetLookupTable())}"
                    out += "\nin-memory size".ljust(15) + f": {varr.GetActualMemorySize()} KB"
            else:
                out += " is empty.\x1b[0m"
            return out

        if self.association == 0:
            out = _get_str(self.obj.dataset.GetPointData(), "Point Data")
        elif self.association == 1:
            out = _get_str(self.obj.dataset.GetCellData(), "Cell Data")
        elif self.association == 2:
            pd = self.obj.dataset.GetFieldData()
            if pd.GetNumberOfArrays():
                out = "\x1b[2m\x1b[1m\x1b[7mMeta Data"
                if self.obj.name:
                    out += f" in {self.obj.name}"
                out += f" contains {pd.GetNumberOfArrays()} entries\x1b[0m"
                for i in range(pd.GetNumberOfArrays()):
                    varr = pd.GetAbstractArray(i)
                    out += f"\n\x1b[1m\x1b[4mEntry name    : {varr.GetName()}\x1b[0m"
                    out += "\nindex".ljust(15) + f": {i}"
                    shape = (varr.GetNumberOfTuples(), varr.GetNumberOfComponents())
                    out += "\nshape".ljust(15) + f": {shape}"

        return out


###############################################################################
class CommonAlgorithms:
    """Common algorithms."""

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
        """Return the size in bytes of the object in memory."""
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
            arr1d = utils.vtk2numpy(cls.dataset.GetLines().GetData())
        except AttributeError:
            return np.array([])
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
            return utils.vtk2numpy(cls.dataset.GetLines().GetData())
        except AttributeError:
            return np.array([])

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
        if not cls.cell_locator:
            cls.cell_locator = vtki.new("CellTreeLocator")
            cls.cell_locator.SetDataSet(cls.dataset)
            cls.cell_locator.BuildLocator()
        cls.cell_locator.FindCellsWithinBounds(bnds, cell_ids)
        cids = []
        for i in range(cell_ids.GetNumberOfIds()):
            cid = cell_ids.GetId(i)
            cids.append(cid)
        return np.array(cids)

    def find_cells_along_line(cls, p0, p1, tol=0.001) -> np.ndarray:
        """
        Find cells that are intersected by a line segment.
        """
        cell_ids = vtki.vtkIdList()
        if not cls.cell_locator:
            cls.cell_locator = vtki.new("CellTreeLocator")
            cls.cell_locator.SetDataSet(cls.dataset)
            cls.cell_locator.BuildLocator()
        cls.cell_locator.FindCellsAlongLine(p0, p1, tol, cell_ids)
        cids = []
        for i in range(cell_ids.GetNumberOfIds()):
            cid = cell_ids.GetId(i)
            cids.append(cid)
        return np.array(cids)

    def find_cells_along_plane(cls, origin, normal, tol=0.001) -> np.ndarray:
        """
        Find cells that are intersected by a plane.
        """
        cell_ids = vtki.vtkIdList()
        if not cls.cell_locator:
            cls.cell_locator = vtki.new("CellTreeLocator")
            cls.cell_locator.SetDataSet(cls.dataset)
            cls.cell_locator.BuildLocator()
        cls.cell_locator.FindCellsAlongPlane(origin, normal, tol, cell_ids)
        cids = []
        for i in range(cell_ids.GetNumberOfIds()):
            cid = cell_ids.GetId(i)
            cids.append(cid)
        return np.array(cids)

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
        """Return the vertices (points) coordinates."""
        try:
            # for polydata and unstructured grid
            varr = cls.dataset.GetPoints().GetData()
        except (AttributeError, TypeError):
            try:
                # for RectilinearGrid, StructuredGrid
                vpts = vtki.vtkPoints()
                cls.dataset.GetPoints(vpts)
                varr = vpts.GetData()
            except (AttributeError, TypeError):
                try:
                    # for ImageData
                    v2p = vtki.new("ImageToPoints")
                    v2p.SetInputData(cls.dataset)
                    v2p.Update()
                    varr = v2p.GetOutput().GetPoints().GetData()
                except AttributeError:
                    return np.array([])

        return utils.vtk2numpy(varr)

    # setter
    @vertices.setter
    def vertices(cls, pts):
        """Set vertices (points) coordinates."""
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
        """Return the vertices (points) coordinates. Same as `vertices`."""
        return cls.vertices

    @points.setter
    def points(cls, pts):
        """Set vertices (points) coordinates. Same as `vertices`."""
        cls.vertices = pts

    @property
    def coordinates(cls):
        """Return the vertices (points) coordinates. Same as `vertices`."""
        return cls.vertices

    @coordinates.setter
    def coordinates(cls, pts):
        """Set vertices (points) coordinates. Same as `vertices`."""
        cls.vertices = pts

    @property
    def cells_as_flat_array(cls):
        """
        Get cell connectivity ids as a 1D numpy array.
        Format is e.g. [3,  10,20,30  4, 10,11,12,13  ...]
        """
        try:
            # valid for unstructured grid
            arr1d = utils.vtk2numpy(cls.dataset.GetCells().GetData())
        except AttributeError:
            # valid for polydata
            arr1d = utils.vtk2numpy(cls.dataset.GetPolys().GetData())
        return arr1d

    @property
    def cells(cls):
        """
        Get the cells connectivity ids as a numpy array.

        The output format is: `[[id0 ... idn], [id0 ... idm],  etc]`.
        """
        try:
            # valid for unstructured grid
            arr1d = utils.vtk2numpy(cls.dataset.GetCells().GetData())
        except AttributeError:
            try:
                # valid for polydata
                arr1d = utils.vtk2numpy(cls.dataset.GetPolys().GetData())
            except AttributeError:
                vedo.logger.warning(f"Cannot get cells for {type(cls)}")
                return np.array([])

        # Get cell connettivity ids as a 1D array. vtk format is:
        # [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        i = 0
        conn = []
        n = len(arr1d)
        if n:
            while True:
                cell = [arr1d[i + k] for k in range(1, arr1d[i] + 1)]
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

        # remove arrays that are already present in the source
        # this is because the interpolator will ignore them otherwise
        for i in range(cls.dataset.GetPointData().GetNumberOfArrays()):
            name = cls.dataset.GetPointData().GetArrayName(i)
            if name not in exclude:
                cls.dataset.GetPointData().RemoveArray(name)

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

        Two new arrays are added to the mesh: `PointID` and `CellID`.
        """
        ids = vtki.new("IdFilter")
        ids.SetInputData(cls.dataset)
        ids.PointIdsOn()
        ids.CellIdsOn()
        ids.FieldDataOff()
        ids.SetPointIdsArrayName("PointID")
        ids.SetCellIdsArrayName("CellID")
        ids.Update()
        cls._update(ids.GetOutput(), reset_locators=False)
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
        gra = vtki.new("GradientFilter")
        if on.startswith("p"):
            varr = cls.dataset.GetPointData()
            tp = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS
        elif on.startswith("c"):
            varr = cls.dataset.GetCellData()
            tp = vtki.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            vedo.logger.error(f"in gradient: unknown option {on}")
            raise RuntimeError

        if input_array is None:
            if varr.GetScalars():
                input_array = varr.GetScalars().GetName()
            else:
                vedo.logger.error(f"in gradient: no scalars found for {on}")
                raise RuntimeError

        gra.SetInputData(cls.dataset)
        gra.SetInputScalars(tp, input_array)
        gra.SetResultArrayName("Gradient")
        gra.SetFasterApproximation(fast)
        gra.ComputeDivergenceOff()
        gra.ComputeVorticityOff()
        gra.ComputeGradientOn()
        gra.Update()
        # cls._update(gra.GetOutput(), reset_locators=False)
        if on.startswith("p"):
            gvecs = utils.vtk2numpy(gra.GetOutput().GetPointData().GetArray("Gradient"))
        else:
            gvecs = utils.vtk2numpy(gra.GetOutput().GetCellData().GetArray("Gradient"))
        return gvecs

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
                that performs fewer derivative calculations (and is therefore faster).
        """
        div = vtki.new("GradientFilter")
        if on.startswith("p"):
            varr = cls.dataset.GetPointData()
            tp = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS
        elif on.startswith("c"):
            varr = cls.dataset.GetCellData()
            tp = vtki.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            vedo.logger.error(f"in divergence(): unknown option {on}")
            raise RuntimeError

        if array_name is None:
            if varr.GetVectors():
                array_name = varr.GetVectors().GetName()
            else:
                vedo.logger.error(f"in divergence(): no vectors found for {on}")
                raise RuntimeError

        div.SetInputData(cls.dataset)
        div.SetInputScalars(tp, array_name)
        div.ComputeDivergenceOn()
        div.ComputeGradientOff()
        div.ComputeVorticityOff()
        div.SetDivergenceArrayName("Divergence")
        div.SetFasterApproximation(fast)
        div.Update()
        # cls._update(div.GetOutput(), reset_locators=False)
        if on.startswith("p"):
            dvecs = utils.vtk2numpy(div.GetOutput().GetPointData().GetArray("Divergence"))
        else:
            dvecs = utils.vtk2numpy(div.GetOutput().GetCellData().GetArray("Divergence"))
        return dvecs

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
        vort = vtki.new("GradientFilter")
        if on.startswith("p"):
            varr = cls.dataset.GetPointData()
            tp = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS
        elif on.startswith("c"):
            varr = cls.dataset.GetCellData()
            tp = vtki.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            vedo.logger.error(f"in vorticity(): unknown option {on}")
            raise RuntimeError

        if array_name is None:
            if varr.GetVectors():
                array_name = varr.GetVectors().GetName()
            else:
                vedo.logger.error(f"in vorticity(): no vectors found for {on}")
                raise RuntimeError

        vort.SetInputData(cls.dataset)
        vort.SetInputScalars(tp, array_name)
        vort.ComputeDivergenceOff()
        vort.ComputeGradientOff()
        vort.ComputeVorticityOn()
        vort.SetVorticityArrayName("Vorticity")
        vort.SetFasterApproximation(fast)
        vort.Update()
        if on.startswith("p"):
            vvecs = utils.vtk2numpy(vort.GetOutput().GetPointData().GetArray("Vorticity"))
        else:
            vvecs = utils.vtk2numpy(vort.GetOutput().GetCellData().GetArray("Vorticity"))
        return vvecs

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
            "write", parents=[cls], comment=filename[:15], shape="folder", c="#8a817c"
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
            Certain data attributes cannot be correctly interpolated. For example, surface normals are expected to be |n|=1;
            after attribute smoothing this constraint is likely to be violated.
            Other vectors and tensors may suffer from similar issues.
            In such a situation, specify `exclude=...` which will not be smoothed (and simply passed through to the output).
            Distance weighting function is based on averaging, 1/r, or 1/r**2 weights, where r is the distance
            between the point to be smoothed and an edge connected neighbor (defined by the smoothing stencil).
            The weights are normalized so that sum(w(i))==1. When smoothing based on averaging, the weights are simply 1/n,
            where n is the number of connected points in the stencil.
            The smoothing process reduces high frequency information in the data attributes.
            With excessive smoothing (large numbers of iterations, and/or a large relaxation factor) important details may be lost,
            and the attributes will move towards an "average" value.
            While this filter will process any dataset type, if the input data is a 3D image volume, it's likely much faster to use
            an image-based algorithm to perform data smoothing.
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
class PointAlgorithms(CommonAlgorithms):
    """Methods for point clouds."""

    def apply_transform(cls, LT: Any, deep_copy=True) -> Self:
        """
        Apply a linear or non-linear transformation to the mesh polygonal data.

        Example:
        ```python
        from vedo import Cube, show, settings
        settings.use_parallel_projection = True
        c1 = Cube().rotate_z(25).pos(2,1).mirror().alpha(0.5)
        T = c1.transform  # rotate by 5 degrees, place at (2,1)
        c2 = Cube().c('red4').wireframe().lw(10).lighting('off')
        c2.apply_transform(T)
        show(c1, c2, "The 2 cubes should overlap!", axes=1).close()
        ```

        ![](https://vedo.embl.es/images/feats/apply_transform.png)
        """
        if cls.dataset.GetNumberOfPoints() == 0:
            return cls

        if isinstance(LT, LinearTransform):
            LT_is_linear = True
            tr = LT.T
            if LT.is_identity():
                return cls

        elif isinstance(LT, (vtki.vtkMatrix4x4, vtki.vtkLinearTransform)) or utils.is_sequence(LT):
            LT_is_linear = True
            LT = LinearTransform(LT)
            tr = LT.T
            if LT.is_identity():
                return cls

        elif isinstance(LT, NonLinearTransform):
            LT_is_linear = False
            tr = LT.T
            cls.transform = LT  # reset

        elif isinstance(LT, vtki.vtkThinPlateSplineTransform):
            LT_is_linear = False
            tr = LT
            cls.transform = NonLinearTransform(LT)  # reset

        else:
            vedo.logger.error(f"apply_transform(), unknown input type:\n{LT}")
            return cls

        ################
        if LT_is_linear:
            try:
                # cls.transform might still not be linear
                cls.transform.concatenate(LT)
            except AttributeError:
                # in that case reset it
                cls.transform = LinearTransform()

        ################
        if isinstance(cls.dataset, vtki.vtkPolyData):
            tp = vtki.new("TransformPolyDataFilter")
        elif isinstance(cls.dataset, vtki.vtkUnstructuredGrid):
            tp = vtki.new("TransformFilter")
            tp.TransformAllInputVectorsOn()
        # elif isinstance(cls.dataset, vtki.vtkImageData):
        #     tp = vtki.new("ImageReslice")
        #     tp.SetInterpolationModeToCubic()
        #     tp.SetResliceTransform(tr)
        else:
            vedo.logger.error(f"apply_transform(), unknown input type: {[cls.dataset]}")
            return cls

        tp.SetTransform(tr)
        tp.SetInputData(cls.dataset)
        tp.Update()
        out = tp.GetOutput()

        if deep_copy:
            cls.dataset.DeepCopy(out)
        else:
            cls.dataset.ShallowCopy(out)

        # reset the locators
        cls.point_locator = None
        cls.cell_locator = None
        cls.line_locator = None
        return cls

    def apply_transform_from_actor(cls) -> LinearTransform:
        """
        Apply the current transformation of the actor to the data.
        Useful when manually moving an actor (eg. when pressing "a").
        Returns the `LinearTransform` object.

        Note that this method is automatically called when the window is closed,
        or the interactor style is changed.
        """
        M = cls.actor.GetMatrix()
        cls.apply_transform(M)
        iden = vtki.vtkMatrix4x4()
        cls.actor.PokeMatrix(iden)
        return LinearTransform(M)

    def pos(cls, x=None, y=None, z=None) -> Self:
        """Set/Get object position."""
        if x is None:  # get functionality
            return cls.transform.position

        if z is None and y is None:  # assume x is of the form (x,y,z)
            if len(x) == 3:
                x, y, z = x
            else:
                x, y = x
                z = 0
        elif z is None:  # assume x,y is of the form x, y
            z = 0

        q = cls.transform.position
        delta = [x, y, z] - q
        if delta[0] == delta[1] == delta[2] == 0:
            return cls
        LT = LinearTransform().translate(delta)
        return cls.apply_transform(LT)

    def shift(cls, dx=0, dy=0, dz=0) -> Self:
        """Add a vector to the current object position."""
        if utils.is_sequence(dx):
            dx, dy, dz = utils.make3d(dx)
        if dx == dy == dz == 0:
            return cls
        LT = LinearTransform().translate([dx, dy, dz])
        return cls.apply_transform(LT)

    def x(cls, val=None) -> Self:
        """Set/Get object position along x axis."""
        p = cls.transform.position
        if val is None:
            return p[0]
        cls.pos(val, p[1], p[2])
        return cls

    def y(cls, val=None)-> Self:
        """Set/Get object position along y axis."""
        p = cls.transform.position
        if val is None:
            return p[1]
        cls.pos(p[0], val, p[2])
        return cls

    def z(cls, val=None) -> Self:
        """Set/Get object position along z axis."""
        p = cls.transform.position
        if val is None:
            return p[2]
        cls.pos(p[0], p[1], val)
        return cls

    def rotate(cls, angle: float, axis=(1, 0, 0), point=(0, 0, 0), rad=False) -> Self:
        """
        Rotate around an arbitrary `axis` passing through `point`.

        Example:
        ```python
        from vedo import *
        c1 = Cube()
        c2 = c1.clone().c('violet').alpha(0.5) # copy of c1
        v = vector(0.2,1,0)
        p = vector(1,0,0)  # axis passes through this point
        c2.rotate(90, axis=v, point=p)
        l = Line(-v+p, v+p).lw(3).c('red')
        show(c1, l, c2, axes=1).close()
        ```

        ![](https://vedo.embl.es/images/feats/rotate_axis.png)
        """
        LT = LinearTransform()
        LT.rotate(angle, axis, point, rad)
        return cls.apply_transform(LT)

    def rotate_x(cls, angle: float, rad=False, around=None) -> Self:
        """
        Rotate around x-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return cls
        LT = LinearTransform().rotate_x(angle, rad, around)
        return cls.apply_transform(LT)

    def rotate_y(cls, angle: float, rad=False, around=None) -> Self:
        """
        Rotate around y-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return cls
        LT = LinearTransform().rotate_y(angle, rad, around)
        return cls.apply_transform(LT)

    def rotate_z(cls, angle: float, rad=False, around=None) -> Self:
        """
        Rotate around z-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return cls
        LT = LinearTransform().rotate_z(angle, rad, around)
        return cls.apply_transform(LT)

    def reorient(cls, initaxis, newaxis, rotation=0, rad=False, xyplane=False) -> Self:
        """
        Reorient the object to point to a new direction from an initial one.
        If `initaxis` is None, the object will be assumed in its "default" orientation.
        If `xyplane` is True, the object will be rotated to lie on the xy plane.

        Use `rotation` to first rotate the object around its `initaxis`.
        """
        q = cls.transform.position
        LT = LinearTransform()
        LT.reorient(initaxis, newaxis, q, rotation, rad, xyplane)
        return cls.apply_transform(LT)

    def scale(cls, s=None, reset=False, origin=True) -> Union[Self, np.array]:
        """
        Set/get object's scaling factor.

        Arguments:
            s : (list, float)
                scaling factor(s).
            reset : (bool)
                if True previous scaling factors are ignored.
            origin : (bool)
                if True scaling is applied with respect to object's position,
                otherwise is applied respect to (0,0,0).

        Note:
            use `s=(sx,sy,sz)` to scale differently in the three coordinates.
        """
        if s is None:
            return np.array(cls.transform.T.GetScale())

        if not utils.is_sequence(s):
            s = [s, s, s]

        LT = LinearTransform()
        if reset:
            old_s = np.array(cls.transform.T.GetScale())
            LT.scale(s / old_s)
        else:
            if origin is True:
                LT.scale(s, origin=cls.transform.position)
            elif origin is False:
                LT.scale(s, origin=False)
            else:
                LT.scale(s, origin=origin)

        return cls.apply_transform(LT)


###############################################################################
class VolumeAlgorithms(CommonAlgorithms):
    """Methods for Volume objects."""

    def bounds(cls) -> np.ndarray:
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        # OVERRIDE CommonAlgorithms.bounds() which is too slow
        return np.array(cls.dataset.GetBounds())

    def isosurface(cls, value=None, flying_edges=False) -> "vedo.mesh.Mesh":
        """
        Return an `Mesh` isosurface extracted from the `Volume` object.

        Set `value` as single float or list of values to draw the isosurface(s).
        Use flying_edges for faster results (but sometimes can interfere with `smooth()`).

        The isosurface values can be accessed with `mesh.metadata["isovalue"]`.

        Examples:
            - [isosurfaces1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces1.py)

                ![](https://vedo.embl.es/images/volumetric/isosurfaces.png)
        """
        scrange = cls.dataset.GetScalarRange()

        if flying_edges:
            cf = vtki.new("FlyingEdges3D")
            cf.InterpolateAttributesOn()
        else:
            cf = vtki.new("ContourFilter")
            cf.UseScalarTreeOn()

        cf.SetInputData(cls.dataset)
        cf.ComputeNormalsOn()

        if utils.is_sequence(value):
            cf.SetNumberOfContours(len(value))
            for i, t in enumerate(value):
                cf.SetValue(i, t)
        else:
            if value is None:
                value = (2 * scrange[0] + scrange[1]) / 3.0
                # print("automatic isosurface value =", value)
            cf.SetValue(0, value)

        cf.Update()
        poly = cf.GetOutput()

        out = vedo.mesh.Mesh(poly, c=None).phong()
        out.mapper.SetScalarRange(scrange[0], scrange[1])
        out.metadata["isovalue"] = value

        out.pipeline = utils.OperationNode(
            "isosurface",
            parents=[cls],
            comment=f"#pts {out.dataset.GetNumberOfPoints()}",
            c="#4cc9f0:#e9c46a",
        )
        return out

    def isosurface_discrete(
            cls, values, background_label=None, internal_boundaries=True, use_quads=False, nsmooth=0,
        ) -> "vedo.mesh.Mesh":
        """
        Create boundary/isocontour surfaces from a label map (e.g., a segmented image) using a threaded,
        3D version of the multiple objects/labels Surface Nets algorithm.
        The input is a 3D image (i.e., volume) where each voxel is labeled
        (integer labels are preferred to real values), and the output data is a polygonal mesh separating
        labeled regions / objects.
        (Note that on output each region [corresponding to a different segmented object] will share
        points/edges on a common boundary, i.e., two neighboring objects will share the boundary that separates them).

        Besides output geometry defining the surface net, the filter outputs a two-component celldata array indicating
        the labels on either side of the polygons composing the output Mesh.
        (This can be used for advanced operations like extracting shared/contacting boundaries between two objects.
        The name of this celldata array is "BoundaryLabels").

        The values can be accessed with `mesh.metadata["isovalue"]`.

        Arguments:
            value : (float, list)
                single value or list of values to draw the isosurface(s).
            background_label : (float)
                this value specifies the label value to use when referencing the background
                region outside of any of the specified regions.
            boundaries : (bool, list)
                if True, the output will only contain the boundary surface. Internal surfaces will be removed.
                If a list of integers is provided, only the boundaries between the specified labels will be extracted.
            use_quads : (bool)
                if True, the output polygons will be quads. If False, the output polygons will be triangles.
            nsmooth : (int)
                number of iterations of smoothing (0 means no smoothing).

        Examples:
            - [isosurfaces2.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces2.py)
        """
        logger = vtki.get_class("Logger")
        logger.SetStderrVerbosity(logger.VERBOSITY_ERROR)

        snets = vtki.new("SurfaceNets3D")
        snets.SetInputData(cls.dataset)

        if nsmooth:
            snets.SmoothingOn()
            snets.AutomaticSmoothingConstraintsOn()
            snets.GetSmoother().SetNumberOfIterations(nsmooth)
            # snets.GetSmoother().SetRelaxationFactor(relaxation_factor)
            # snets.GetSmoother().SetConstraintDistance(constraint_distance)
        else:
            snets.SmoothingOff()

        if internal_boundaries is False:
            snets.SetOutputStyleToBoundary()
        elif internal_boundaries is True:
            snets.SetOutputStyleToDefault()
        elif utils.is_sequence(internal_boundaries):
            snets.SetOutputStyleToSelected()
            snets.InitializeSelectedLabelsList()
            for val in internal_boundaries:
                snets.AddSelectedLabel(val)
        else:
            vedo.logger.error("isosurface_discrete(): unknown boundaries option")

        n = len(values)
        snets.SetNumberOfContours(n)
        snets.SetNumberOfLabels(n)

        if background_label is not None:
            snets.SetBackgroundLabel(background_label)

        for i, val in enumerate(values):
            snets.SetValue(i, val)

        if use_quads:
            snets.SetOutputMeshTypeToQuads()
        else:
            snets.SetOutputMeshTypeToTriangles()
        snets.Update()

        out = vedo.mesh.Mesh(snets.GetOutput())
        out.metadata["isovalue"] = values
        out.pipeline = utils.OperationNode(
            "isosurface_discrete",
            parents=[cls],
            comment=f"#pts {out.dataset.GetNumberOfPoints()}",
            c="#4cc9f0:#e9c46a",
        )

        logger.SetStderrVerbosity(logger.VERBOSITY_INFO)
        return out


    def legosurface(
        cls,
        vmin=None,
        vmax=None,
        invert=False,
        boundary=True,
        array_name="input_scalars",
    ) -> "vedo.mesh.Mesh":
        """
        Represent an object - typically a `Volume` - as lego blocks (voxels).
        By default colors correspond to the volume's scalar.
        Returns an `Mesh` object.

        Arguments:
            vmin : (float)
                the lower threshold, voxels below this value are not shown.
            vmax : (float)
                the upper threshold, voxels above this value are not shown.
            boundary : (bool)
                controls whether to include cells that are partially inside
            array_name : (int, str)
                name or index of the scalar array to be considered

        Examples:
            - [legosurface.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/legosurface.py)

                ![](https://vedo.embl.es/images/volumetric/56820682-da40e500-684c-11e9-8ea3-91cbcba24b3a.png)
        """
        imp_dataset = vtki.new("ImplicitDataSet")
        imp_dataset.SetDataSet(cls.dataset)
        window = vtki.new("ImplicitWindowFunction")
        window.SetImplicitFunction(imp_dataset)

        srng = list(cls.dataset.GetScalarRange())
        if vmin is not None:
            srng[0] = vmin
        if vmax is not None:
            srng[1] = vmax
        if not boundary:
            tol = 0.00001 * (srng[1] - srng[0])
            srng[0] -= tol
            srng[1] += tol
        window.SetWindowRange(srng)
        # print("legosurface window range:", srng)

        extract = vtki.new("ExtractGeometry")
        extract.SetInputData(cls.dataset)
        extract.SetImplicitFunction(window)
        extract.SetExtractInside(invert)
        extract.SetExtractBoundaryCells(boundary)
        extract.Update()

        gf = vtki.new("GeometryFilter")
        gf.SetInputData(extract.GetOutput())
        gf.Update()

        m = vedo.mesh.Mesh(gf.GetOutput()).lw(0.1).flat()
        m.map_points_to_cells()
        m.celldata.select(array_name)

        m.pipeline = utils.OperationNode(
            "legosurface",
            parents=[cls],
            comment=f"array: {array_name}",
            c="#4cc9f0:#e9c46a",
        )
        return m

    def tomesh(cls, fill=True, shrink=1.0) -> "vedo.mesh.Mesh":
        """
        Build a polygonal Mesh from the current object.

        If `fill=True`, the interior faces of all the cells are created.
        (setting a `shrink` value slightly smaller than the default 1.0
        can avoid flickering due to internal adjacent faces).

        If `fill=False`, only the boundary faces will be generated.
        """
        gf = vtki.new("GeometryFilter")
        if fill:
            sf = vtki.new("ShrinkFilter")
            sf.SetInputData(cls.dataset)
            sf.SetShrinkFactor(shrink)
            sf.Update()
            gf.SetInputData(sf.GetOutput())
            gf.Update()
            poly = gf.GetOutput()
            if shrink == 1.0:
                clean_poly = vtki.new("CleanPolyData")
                clean_poly.PointMergingOn()
                clean_poly.ConvertLinesToPointsOn()
                clean_poly.ConvertPolysToLinesOn()
                clean_poly.ConvertStripsToPolysOn()
                clean_poly.SetInputData(poly)
                clean_poly.Update()
                poly = clean_poly.GetOutput()
        else:
            gf.SetInputData(cls.dataset)
            gf.Update()
            poly = gf.GetOutput()

        msh = vedo.mesh.Mesh(poly).flat()
        msh.scalarbar = cls.scalarbar
        lut = utils.ctf2lut(cls)
        if lut:
            msh.mapper.SetLookupTable(lut)

        msh.pipeline = utils.OperationNode(
            "tomesh", parents=[cls], comment=f"fill={fill}", c="#9e2a2b:#e9c46a"
        )
        return msh
