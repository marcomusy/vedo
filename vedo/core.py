#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

warnings = dict(
    points_getter=(
        "WARNING: points() is deprecated, use vertices instead. Change:\n"
        "         mesh.points() -> mesh.vertices\n"
        "         (silence this with vedo.core.warnings['points_getter']=False)"
    ),
    points_setter=(
        "WARNING: points() is deprecated, use vertices instead. Change:\n"
        "         mesh.points([[x,y,z], ...]) -> mesh.vertices = [[x,y,z], ...]\n"
        "         (silence this with vedo.core.warnings['points_getter']=False)"
    ),
)

###############################################################################
class DataArrayHelper:
    # Internal use only.
    # Helper class to manage data associated to either
    # points (or vertices) and cells (or faces).
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

    def keys(self):
        """Return the list of available data array names"""
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
                arrnames.append(name)
        return arrnames

    def items(self):
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

    def todict(self):
        """Return a dictionary of the available data arrays."""
        return dict(self.items())

    def rename(self, oldname, newname):
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

    def remove(self, key):
        """Remove a data array by name or number"""
        if self.association == 0:
            self.obj.dataset.GetPointData().RemoveArray(key)
        elif self.association == 1:
            self.obj.dataset.GetCellData().RemoveArray(key)
        elif self.association == 2:
            self.obj.dataset.GetFieldData().RemoveArray(key)

    def clear(self):
        """Remove all data associated to this object"""
        if self.association == 0:
            data = self.obj.dataset.GetPointData()
        elif self.association == 1:
            data = self.obj.dataset.GetCellData()
        elif self.association == 2:
            data = self.obj.dataset.GetFieldData()
        for i in range(data.GetNumberOfArrays()):
            if self.association == 2:
                name = data.GetAbstractArray(i).GetName()
            else:
                name = data.GetArray(i).GetName()
            data.RemoveArray(name)

    def select(self, key):
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
            if "rgb" in key.lower():
                data.SetActiveScalars(key)
                try:
                    # could be a volume mapper
                    self.obj.mapper.SetColorModeToDirectScalars()
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

    def select_texture_coords(self, key):
        """Select one specific array to be used as texture coordinates."""
        if self.association == 0:
            data = self.obj.dataset.GetPointData()
        else:
            vedo.logger.warning("texture coordinates are only available for point data")
            return

        if isinstance(key, int):
            key = data.GetArrayName(key)
        data.SetTCoords(data.GetArray(key))
        return self.obj

    def select_normals(self, key):
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

    def print(self, **kwargs):
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
                    if t in vedo.utils.array_types:
                        out += "\ntype".ljust(15)
                        out += f": {vedo.utils.array_types[t][1]} ({vedo.utils.array_types[t][0]})"
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
    def pointdata(self):
        """
        Create and/or return a `numpy.array` associated to points (vertices).
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.pointdata["arrayname"]`

        Usage:

            `myobj.pointdata.keys()` to return the available data array names

            `myobj.pointdata.select(name)` to make this array the active one

            `myobj.pointdata.remove(name)` to remove this array
        """
        return DataArrayHelper(self, 0)

    @property
    def celldata(self):
        """
        Create and/or return a `numpy.array` associated to cells (faces).
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.celldata["arrayname"]`

        Usage:

            `myobj.celldata.keys()` to return the available data array names

            `myobj.celldata.select(name)` to make this array the active one

            `myobj.celldata.remove(name)` to remove this array
        """
        return DataArrayHelper(self, 1)

    @property
    def metadata(self):
        """
        Create and/or return a `numpy.array` associated to neither cells nor faces.
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.metadata["arrayname"]`

        Usage:

            `myobj.metadata.keys()` to return the available data array names

            `myobj.metadata.select(name)` to make this array the active one

            `myobj.metadata.remove(name)` to remove this array
        """
        return DataArrayHelper(self, 2)

    def memory_address(self):
        """
        Return a unique memory address integer which may serve as the ID of the
        object, or passed to c++ code.
        """
        # https://www.linkedin.com/pulse/speedup-your-code-accessing-python-vtk-objects-from-c-pletzer/
        # https://github.com/tfmoraes/polydata_connectivity
        return int(self.dataset.GetAddressAsString("")[5:], 16)

    def memory_size(self):
        """Return the size in bytes of the object in memory."""
        return self.dataset.GetActualMemorySize()

    def modified(self):
        """Use in conjunction with `tonumpy()` to update any modifications to the image array."""
        self.dataset.GetPointData().Modified()
        scals = self.dataset.GetPointData().GetScalars()
        if scals:
            scals.Modified()
        return self

    def box(self, scale=1, padding=0):
        """
        Return the bounding box as a new `Mesh` object.

        Arguments:
            scale : (float)
                box size can be scaled by a factor
            padding : (float, list)
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
            width  * scale + padding[1],
            height * scale + padding[2],
            c="gray",
        )
        try:
            pr = vtki.vtkProperty()
            pr.DeepCopy(self.properties)
            bx.SetProperty(pr)
            bx.properties = pr
        except (AttributeError, TypeError):
            pass
        bx.flat().lighting("off").wireframe(True)
        return bx
    
    def update_dataset(self, dataset, **kwargs):
        """Update the dataset of the object with the provided VTK dataset."""
        self._update(dataset, **kwargs)
        return self

    def bounds(self):
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        try:  # this is very slow for large meshes
            pts = self.vertices
            xmin, ymin, zmin = np.min(pts, axis=0)
            xmax, ymax, zmax = np.max(pts, axis=0)
            return np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        except (AttributeError, ValueError):
            return np.array(self.dataset.GetBounds())

    def xbounds(self, i=None):
        """Get the bounds `[xmin,xmax]`. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i is not None:
            return b[i]
        return np.array([b[0], b[1]])

    def ybounds(self, i=None):
        """Get the bounds `[ymin,ymax]`. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i == 0:
            return b[2]
        if i == 1:
            return b[3]
        return np.array([b[2], b[3]])

    def zbounds(self, i=None):
        """Get the bounds `[zmin,zmax]`. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i == 0:
            return b[4]
        if i == 1:
            return b[5]
        return np.array([b[4], b[5]])

    def diagonal_size(self):
        """Get the length of the diagonal of the bounding box."""
        b = self.bounds()
        return np.sqrt((b[1] - b[0])**2 + (b[3] - b[2])**2 + (b[5] - b[4])**2)

    def average_size(self):
        """
        Calculate and return the average size of the object.
        This is the mean of the vertex distances from the center of mass.
        """
        coords = self.vertices
        cm = np.mean(coords, axis=0)
        if coords.shape[0] == 0:
            return 0.0
        cc = coords - cm
        return np.mean(np.linalg.norm(cc, axis=1))

    def center_of_mass(self):
        """Get the center of mass of the object."""
        if isinstance(self, (vedo.RectilinearGrid, vedo.Volume)):
            return np.array(self.dataset.GetCenter())
        cmf = vtki.new("CenterOfMass")
        cmf.SetInputData(self.dataset)
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)

    def copy_data_from(self, obj):
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
        colors.printc("WARNING: 'inputdata()' is obsolete, use '.dataset' instead.", c="y")
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

    def points(self, pts=None):
        """Obsolete, use `self.vertices` or `self.coordinates` instead."""
        if pts is None:  ### getter

            if warnings["points_getter"]:
                colors.printc(warnings["points_getter"], c="y")
                warnings["points_getter"] = ""
            return self.vertices

        else:  ### setter

            if warnings["points_setter"]:
                colors.printc(warnings["points_setter"], c="y")
                warnings["points_setter"] = ""

            pts = np.asarray(pts, dtype=np.float32)

            if pts.ndim == 1:
                ### getter by point index ###################
                indices = pts.astype(int)
                vpts = self.dataset.GetPoints()
                arr = utils.vtk2numpy(vpts.GetData())
                return arr[indices]  ###########

            ### setter ####################################
            if pts.shape[1] == 2:
                pts = np.c_[pts, np.zeros(pts.shape[0], dtype=np.float32)]
            arr = utils.numpy2vtk(pts, dtype=np.float32)

            vpts = self.dataset.GetPoints()
            vpts.SetData(arr)
            vpts.Modified()
            # reset mesh to identity matrix position/rotation:
            self.point_locator = None
            self.cell_locator = None
            self.line_locator = None
            self.transform = LinearTransform()
            return self

    @property
    def cell_centers(self):
        """
        Get the coordinates of the cell centers.

        Examples:
            - [delaunay2d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/delaunay2d.py)
        
        See also: `CellCenters()`.
        """
        vcen = vtki.new("CellCenters")
        vcen.CopyArraysOff()
        vcen.SetInputData(self.dataset)
        vcen.Update()
        return utils.vtk2numpy(vcen.GetOutput().GetPoints().GetData())

    @property
    def lines(self):
        """
        Get lines connectivity ids as a python array
        formatted as `[[id0,id1], [id3,id4], ...]`

        See also: `lines_as_flat_array()`.
        """
        # Get cell connettivity ids as a 1D array. The vtk format is:
        #    [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        try:
            arr1d = utils.vtk2numpy(self.dataset.GetLines().GetData())
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
    def lines_as_flat_array(self):
        """
        Get lines connectivity ids as a 1D numpy array.
        Format is e.g. [2,  10,20,  3, 10,11,12,  2, 70,80, ...]

        See also: `lines()`.
        """
        try:
            return utils.vtk2numpy(self.dataset.GetLines().GetData())
        except AttributeError:
            return np.array([])

    def mark_boundaries(self):
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

    def find_cells_in_bounds(self, xbounds=(), ybounds=(), zbounds=()):
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
        if not self.cell_locator:
            self.cell_locator = vtki.new("CellTreeLocator")
            self.cell_locator.SetDataSet(self.dataset)
            self.cell_locator.BuildLocator()
        self.cell_locator.FindCellsWithinBounds(bnds, cell_ids)
        cids = []
        for i in range(cell_ids.GetNumberOfIds()):
            cid = cell_ids.GetId(i)
            cids.append(cid)
        return np.array(cids)

    def find_cells_along_line(self, p0, p1, tol=0.001):
        """
        Find cells that are intersected by a line segment.
        """
        cell_ids = vtki.vtkIdList()
        if not self.cell_locator:
            self.cell_locator = vtki.new("CellTreeLocator")
            self.cell_locator.SetDataSet(self.dataset)
            self.cell_locator.BuildLocator()
        self.cell_locator.FindCellsAlongLine(p0, p1, tol, cell_ids)
        cids = []
        for i in range(cell_ids.GetNumberOfIds()):
            cid = cell_ids.GetId(i)
            cids.append(cid)
        return np.array(cids)

    def find_cells_along_plane(self, origin, normal, tol=0.001):
        """
        Find cells that are intersected by a plane.
        """
        cell_ids = vtki.vtkIdList()
        if not self.cell_locator:
            self.cell_locator = vtki.new("CellTreeLocator")
            self.cell_locator.SetDataSet(self.dataset)
            self.cell_locator.BuildLocator()
        self.cell_locator.FindCellsAlongPlane(origin, normal, tol, cell_ids)
        cids = []
        for i in range(cell_ids.GetNumberOfIds()):
            cid = cell_ids.GetId(i)
            cids.append(cid)
        return np.array(cids)

    def map_cells_to_points(self, arrays=(), move=False):
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

    @property
    def vertices(self):
        """Return the vertices (points) coordinates."""
        try:
            # for polydata and unstructured grid
            varr = self.dataset.GetPoints().GetData()
        except (AttributeError, TypeError):
            try:
                # for RectilinearGrid, StructuredGrid
                vpts = vtki.vtkPoints()
                self.dataset.GetPoints(vpts)
                varr = vpts.GetData()
            except (AttributeError, TypeError):
                try:
                    # for ImageData
                    v2p = vtki.new("ImageToPoints")
                    v2p.SetInputData(self.dataset)
                    v2p.Update()
                    varr = v2p.GetOutput().GetPoints().GetData()
                except AttributeError:
                    return np.array([])

        return utils.vtk2numpy(varr)

    # setter
    @vertices.setter
    def vertices(self, pts):
        """Set vertices (points) coordinates."""
        pts = utils.make3d(pts)
        arr = utils.numpy2vtk(pts, dtype=np.float32)
        try:
            vpts = self.dataset.GetPoints()
            vpts.SetData(arr)
            vpts.Modified()
        except (AttributeError, TypeError):
            vedo.logger.error(f"Cannot set vertices for {type(self)}")
            return self
        # reset mesh to identity matrix position/rotation:
        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        self.transform = LinearTransform()

    @property
    def coordinates(self):
        """Return the vertices (points) coordinates. Same as `vertices`."""
        return self.vertices

    @coordinates.setter
    def coordinates(self, pts):
        """Set vertices (points) coordinates. Same as `vertices`."""
        self.vertices = pts

    @property
    def cells_as_flat_array(self):
        """
        Get cell connectivity ids as a 1D numpy array.
        Format is e.g. [3,  10,20,30  4, 10,11,12,13  ...]
        """
        try:
            # valid for unstructured grid
            arr1d = utils.vtk2numpy(self.dataset.GetCells().GetData())
        except AttributeError:
            # valid for polydata
            arr1d = utils.vtk2numpy(self.dataset.GetPolys().GetData())
        return arr1d

    @property
    def cells(self):
        """
        Get the cells connectivity ids as a numpy array.

        The output format is: `[[id0 ... idn], [id0 ... idm],  etc]`.
        """
        try:
            # valid for unstructured grid
            arr1d = utils.vtk2numpy(self.dataset.GetCells().GetData())
        except AttributeError:
            try:
                # valid for polydata
                arr1d = utils.vtk2numpy(self.dataset.GetPolys().GetData())
            except AttributeError:
                vedo.logger.warning(f"Cannot get cells for {type(self)}")
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

    def map_points_to_cells(self, arrays=(), move=False):
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

    def resample_data_from(self, source, tol=None, categorical=False):
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
        pts = m1.vertices
        ces = m1.cell_centers
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
        if tol:
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
    ):
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
        interpolator.Update()

        if on == "cells":
            p2c = vtki.new("PointDataToCellData")
            p2c.SetInputData(interpolator.GetOutput())
            p2c.Update()
            cpoly = p2c.GetOutput()
        else:
            cpoly = interpolator.GetOutput()

        self._update(cpoly, reset_locators=False)

        self.pipeline = utils.OperationNode("interpolate_data_from", parents=[self, source])
        return self

    def add_ids(self):
        """
        Generate point and cell ids arrays.

        Two new arrays are added to the mesh: `PointID` and `CellID`.
        """
        ids = vtki.new("IdFilter")
        ids.SetInputData(self.dataset)
        ids.PointIdsOn()
        ids.CellIdsOn()
        ids.FieldDataOff()
        ids.SetPointIdsArrayName("PointID")
        ids.SetCellIdsArrayName("CellID")
        ids.Update()
        self._update(ids.GetOutput(), reset_locators=False)
        self.pipeline = utils.OperationNode("add_ids", parents=[self])
        return self

    def gradient(self, input_array=None, on="points", fast=False):
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
            varr = self.dataset.GetPointData()
            tp = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS
        elif on.startswith("c"):
            varr = self.dataset.GetCellData()
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

        gra.SetInputData(self.dataset)
        gra.SetInputScalars(tp, input_array)
        gra.SetResultArrayName("Gradient")
        gra.SetFasterApproximation(fast)
        gra.ComputeDivergenceOff()
        gra.ComputeVorticityOff()
        gra.ComputeGradientOn()
        gra.Update()
        if on.startswith("p"):
            gvecs = utils.vtk2numpy(gra.GetOutput().GetPointData().GetArray("Gradient"))
        else:
            gvecs = utils.vtk2numpy(gra.GetOutput().GetCellData().GetArray("Gradient"))
        return gvecs

    def divergence(self, array_name=None, on="points", fast=False):
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
            varr = self.dataset.GetPointData()
            tp = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS
        elif on.startswith("c"):
            varr = self.dataset.GetCellData()
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

        div.SetInputData(self.dataset)
        div.SetInputScalars(tp, array_name)
        div.ComputeDivergenceOn()
        div.ComputeGradientOff()
        div.ComputeVorticityOff()
        div.SetDivergenceArrayName("Divergence")
        div.SetFasterApproximation(fast)
        div.Update()
        if on.startswith("p"):
            dvecs = utils.vtk2numpy(div.GetOutput().GetPointData().GetArray("Divergence"))
        else:
            dvecs = utils.vtk2numpy(div.GetOutput().GetCellData().GetArray("Divergence"))
        return dvecs

    def vorticity(self, array_name=None, on="points", fast=False):
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
            varr = self.dataset.GetPointData()
            tp = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS
        elif on.startswith("c"):
            varr = self.dataset.GetCellData()
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

        vort.SetInputData(self.dataset)
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

    def probe(self, source):
        """
        Takes a data set and probes its scalars at the specified points in space.

        Note that a mask is also output with valid/invalid points which can be accessed
        with `mesh.pointdata['ValidPointMask']`.

        Check out also:
            `interpolate_data_from()`

        Examples:
            - [probe_points.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/probe_points.py)

                ![](https://vedo.embl.es/images/volumetric/probePoints.png)
        """
        probe_filter = vtki.new("ProbeFilter")
        probe_filter.SetSourceData(source.dataset)
        probe_filter.SetInputData(self.dataset)
        probe_filter.Update()
        self._update(probe_filter.GetOutput(), reset_locators=False)
        self.pipeline = utils.OperationNode("probe", parents=[self, source])
        self.pointdata.rename("vtkValidPointMask", "ValidPointMask")
        return self

    def compute_cell_size(self):
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
        return self

    def integrate_data(self):
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

    def write(self, filename, binary=True):
        """Write object to file."""
        out = vedo.file_io.write(self, filename, binary)
        out.pipeline = utils.OperationNode(
            "write", parents=[self], comment=filename[:15], shape="folder", c="#8a817c"
        )
        return out

    def tomesh(self, bounds=(), shrink=0):
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

    def smooth_data(self, 
            niter=10, relaxation_factor=0.1, strategy=0, mask=None,
            exclude=("Normals", "TextureCoordinates"),
        ):
        """
        Smooth point attribute data using distance weighted Laplacian kernel.

        The effect is to blur regions of high variation and emphasize low variation regions.

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
            exclude : (list)
                list of arrays to be excluded from smoothing
        """
        saf = vtki.new("AttributeSmoothingFilter")
        saf.SetInputData(self.dataset)
        saf.SetRelaxationFactor(relaxation_factor)
        saf.SetNumberOfIterations(niter)

        for ex in exclude:
            saf.AddExcludedArray(ex)

        saf.SetWeightsTypeToDistance2()

        saf.SetSmoothingStrategy(strategy)
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

        saf.Update()

        self._update(saf.GetOutput())
        self.pipeline = utils.OperationNode(
            "smooth_data", comment=f"strategy {strategy}", parents=[self], c="#9e2a2b"
        )
        return self
        
    def compute_streamlines(
            self, 
            seeds, 
            integrator="rk4",
            direction="forward",
            initial_step_size=None,
            max_propagation=None,
            max_steps=10000,
            step_length=0,
            surface_constrained=False,
            compute_vorticity=False,
        ):
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
        b = self.dataset.GetBounds()
        size = (b[5]-b[4] + b[3]-b[2] + b[1]-b[0]) / 3
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
        self.pipeline = utils.OperationNode(
            "compute_streamlines", comment=f"{integrator}", parents=[self, seeds], c="#9e2a2b"
        )
        return stlines

###############################################################################
class PointAlgorithms(CommonAlgorithms):
    """Methods for point clouds."""

    def apply_transform(self, LT, concatenate=True, deep_copy=True):
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
        if self.dataset.GetNumberOfPoints() == 0:
            return self

        if isinstance(LT, LinearTransform):
            LT_is_linear = True
            tr = LT.T
            if LT.is_identity():
                return self
        
        elif isinstance(LT, (vtki.vtkMatrix4x4, vtki.vtkLinearTransform)) or utils.is_sequence(LT):
            LT_is_linear = True
            LT = LinearTransform(LT)
            tr = LT.T
            if LT.is_identity():
                return self

        elif isinstance(LT, NonLinearTransform):
            LT_is_linear = False
            tr = LT.T
            self.transform = LT  # reset

        elif isinstance(LT, vtki.vtkThinPlateSplineTransform):
            LT_is_linear = False
            tr = LT
            self.transform = NonLinearTransform(LT)  # reset

        else:
            vedo.logger.error(f"apply_transform(), unknown input type:\n{LT}")
            return self

        ################
        if LT_is_linear:
            if concatenate:
                try:
                    # self.transform might still not be linear
                    self.transform.concatenate(LT)
                except AttributeError:
                    # in that case reset it
                    self.transform = LinearTransform()

        ################
        if isinstance(self.dataset, vtki.vtkPolyData):
            tp = vtki.new("TransformPolyDataFilter")
        elif isinstance(self.dataset, vtki.vtkUnstructuredGrid):
            tp = vtki.new("TransformFilter")
            tp.TransformAllInputVectorsOn()
        # elif isinstance(self.dataset, vtki.vtkImageData):
        #     tp = vtki.new("ImageReslice")
        #     tp.SetInterpolationModeToCubic()
        #     tp.SetResliceTransform(tr)
        else:
            vedo.logger.error(f"apply_transform(), unknown input type: {[self.dataset]}")
            return self
        tp.SetTransform(tr)
        tp.SetInputData(self.dataset)
        tp.Update()
        out = tp.GetOutput()

        if deep_copy:
            self.dataset.DeepCopy(out)
        else:
            self.dataset.ShallowCopy(out)

        # reset the locators
        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        return self

    def apply_transform_from_actor(self):
        """
        Apply the current transformation of the actor to the data.
        Useful when manually moving an actor (eg. when pressing "a").
        Returns the `LinearTransform` object.

        Note that this method is automatically called when the window is closed,
        or the interactor style is changed.
        """
        M = self.actor.GetMatrix()
        self.apply_transform(M)
        iden = vtki.vtkMatrix4x4()
        self.actor.PokeMatrix(iden)
        return LinearTransform(M)

    def pos(self, x=None, y=None, z=None):
        """Set/Get object position."""
        if x is None:  # get functionality
            return self.transform.position

        if z is None and y is None:  # assume x is of the form (x,y,z)
            if len(x) == 3:
                x, y, z = x
            else:
                x, y = x
                z = 0
        elif z is None:  # assume x,y is of the form x, y
            z = 0

        q = self.transform.position
        delta = [x, y, z] - q
        if delta[0] == delta[1] == delta[2] == 0:
            return self
        LT = LinearTransform().translate(delta)
        return self.apply_transform(LT)

    def shift(self, dx=0, dy=0, dz=0):
        """Add a vector to the current object position."""
        if utils.is_sequence(dx):
            dx, dy, dz = utils.make3d(dx)
        if dx == dy == dz == 0:
            return self
        LT = LinearTransform().translate([dx, dy, dz])
        return self.apply_transform(LT)

    def x(self, val=None):
        """Set/Get object position along x axis."""
        p = self.transform.position
        if val is None:
            return p[0]
        self.pos(val, p[1], p[2])
        return self

    def y(self, val=None):
        """Set/Get object position along y axis."""
        p = self.transform.position
        if val is None:
            return p[1]
        self.pos(p[0], val, p[2])
        return self

    def z(self, val=None):
        """Set/Get object position along z axis."""
        p = self.transform.position
        if val is None:
            return p[2]
        self.pos(p[0], p[1], val)
        return self

    def rotate(self, angle, axis=(1, 0, 0), point=(0, 0, 0), rad=False):
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
        return self.apply_transform(LT)

    def rotate_x(self, angle, rad=False, around=None):
        """
        Rotate around x-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return self
        LT = LinearTransform().rotate_x(angle, rad, around)
        return self.apply_transform(LT)

    def rotate_y(self, angle, rad=False, around=None):
        """
        Rotate around y-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return self
        LT = LinearTransform().rotate_y(angle, rad, around)
        return self.apply_transform(LT)

    def rotate_z(self, angle, rad=False, around=None):
        """
        Rotate around z-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return self
        LT = LinearTransform().rotate_z(angle, rad, around)
        return self.apply_transform(LT)

    def reorient(self, initaxis, newaxis, rotation=0, rad=False, xyplane=False):
        """
        Reorient the object to point to a new direction from an initial one.
        If `initaxis` is None, the object will be assumed in its "default" orientation.
        If `xyplane` is True, the object will be rotated to lie on the xy plane.

        Use `rotation` to first rotate the object around its `initaxis`.
        """
        q = self.transform.position
        LT = LinearTransform()
        LT.reorient(initaxis, newaxis, q, rotation, rad, xyplane)
        return self.apply_transform(LT)

    def scale(self, s=None, reset=False, origin=True):
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
            return np.array(self.transform.T.GetScale())

        if not utils.is_sequence(s):
            s = [s, s, s]

        LT = LinearTransform()
        if reset:
            old_s = np.array(self.transform.T.GetScale())
            LT.scale(s / old_s)
        else:
            if origin is True:
                LT.scale(s, origin=self.transform.position)
            elif origin is False:
                LT.scale(s, origin=False)
            else:
                LT.scale(s, origin=origin)

        return self.apply_transform(LT)


###############################################################################
class VolumeAlgorithms(CommonAlgorithms):
    """Methods for Volume objects."""

    def bounds(self):
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        # OVERRIDE CommonAlgorithms.bounds() which is too slow
        return np.array(self.dataset.GetBounds())

    def isosurface(self, value=None, flying_edges=False):
        """
        Return an `Mesh` isosurface extracted from the `Volume` object.

        Set `value` as single float or list of values to draw the isosurface(s).
        Use flying_edges for faster results (but sometimes can interfere with `smooth()`).

        Examples:
            - [isosurfaces1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces1.py)

                ![](https://vedo.embl.es/images/volumetric/isosurfaces.png)
        """
        scrange = self.dataset.GetScalarRange()

        if flying_edges:
            cf = vtki.new("FlyingEdges3D")
            cf.InterpolateAttributesOn()
        else:
            cf = vtki.new("ContourFilter")
            cf.UseScalarTreeOn()

        cf.SetInputData(self.dataset)
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

        out.pipeline = utils.OperationNode(
            "isosurface",
            parents=[self],
            comment=f"#pts {out.dataset.GetNumberOfPoints()}",
            c="#4cc9f0:#e9c46a",
        )
        return out
    
    def isosurface_discrete(self, value=None, nsmooth=15):
        """
        Create boundary/isocontour surfaces from a label map (e.g., a segmented image) using a threaded,
        3D version of the multiple objects/labels Surface Nets algorithm.
        The input is a 3D image (i.e., volume) where each voxel is labeled
        (integer labels are preferred to real values), and the output data is a polygonal mesh separating
        labeled regions / objects.
        (Note that on output each region [corresponding to a different segmented object] will share
        points/edges on a common boundary, i.e., two neighboring objects will share the boundary that separates them).

        Arguments:
            value : (float, list)
                single value or list of values to draw the isosurface(s).
            nsmooth : (int)
                number of iterations of smoothing (0 means no smoothing).

        Examples:
            - [isosurfaces2.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces2.py)
        """
        if not utils.is_sequence(value):
            value = [value]
        
        snets = vtki.new("SurfaceNets3D")
        snets.SetInputData(self.dataset)

        if nsmooth:
            snets.SmoothingOn()
            snets.AutomaticSmoothingConstraintsOn()
            snets.GetSmoother().SetNumberOfIterations(nsmooth)
            # snets.GetSmoother().SetRelaxationFactor(relaxation_factor)
            # snets.GetSmoother().SetConstraintDistance(constraint_distance)
        else:
            snets.SmoothingOff()

        for i, val in enumerate(value):
            snets.SetValue(i, val)
        snets.Update()
        snets.SetOutputMeshTypeToTriangles()
        snets.SetOutputStyleToBoundary()
        snets.Update()

        out = vedo.mesh.Mesh(snets.GetOutput())
        out.pipeline = utils.OperationNode(
            "isosurface_discrete",
            parents=[self],
            comment=f"#pts {out.dataset.GetNumberOfPoints()}",
            c="#4cc9f0:#e9c46a",
        )
        return out


    def legosurface(
        self,
        vmin=None,
        vmax=None,
        invert=False,
        boundary=False,
        array_name="input_scalars",
    ):
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
        imp_dataset.SetDataSet(self.dataset)
        window = vtki.new("ImplicitWindowFunction")
        window.SetImplicitFunction(imp_dataset)

        srng = list(self.dataset.GetScalarRange())
        if vmin is not None:
            srng[0] = vmin
        if vmax is not None:
            srng[1] = vmax
        tol = 0.00001 * (srng[1] - srng[0])
        srng[0] -= tol
        srng[1] += tol
        window.SetWindowRange(srng)

        extract = vtki.new("ExtractGeometry")
        extract.SetInputData(self.dataset)
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
            parents=[self],
            comment=f"array: {array_name}",
            c="#4cc9f0:#e9c46a",
        )
        return m

    def tomesh(self, fill=True, shrink=1.0):
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
            sf.SetInputData(self.dataset)
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
            gf.SetInputData(self.dataset)
            gf.Update()
            poly = gf.GetOutput()

        msh = vedo.mesh.Mesh(poly).flat()
        msh.scalarbar = self.scalarbar
        lut = utils.ctf2lut(self)
        if lut:
            msh.mapper.SetLookupTable(lut)

        msh.pipeline = utils.OperationNode(
            "tomesh", parents=[self], comment=f"fill={fill}", c="#9e2a2b:#e9c46a"
        )
        return msh
