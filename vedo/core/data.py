#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Data-array helper utilities for vedo objects."""

from typing import Any
import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils

__all__ = ["DataArrayHelper", "_get_data_legacy_format"]

def _get_data_legacy_format(arr):
    # try the old way then the new way to get the array in legacy format 
    # #issue #1292
    if utils.vtk_version_at_least(6, 0):
        varr = vtki.vtkIdTypeArray()
        arr.ExportLegacyFormat(varr)
        arr1d = utils.vtk2numpy(varr)
        # print("got legacy format with ExportLegacyFormat", [arr])
    else:
        arr1d = utils.vtk2numpy(arr.GetData())
    return arr1d


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

    def keys(self) -> list[str]:
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

    def items(self) -> list:
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

    def remove(self, key: int | str) -> None:
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
        names = []
        for i in range(data.GetNumberOfArrays()):
            if self.association == 2:
                arr = data.GetAbstractArray(i)
            else:
                arr = data.GetArray(i)
            if arr:
                names.append(arr.GetName())
        for name in names:
            data.RemoveArray(name)

    def select(self, key: int | str) -> Any:
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

    def select_texture_coords(self, key: int | str) -> Any:
        """Select one specific array to be used as texture coordinates."""
        if self.association == 0:
            data = self.obj.dataset.GetPointData()
            if isinstance(key, int):
                key = data.GetArrayName(key)
            data.SetTCoords(data.GetArray(key))
        else:
            vedo.logger.warning("texture coordinates are only available for point data")
        return self.obj

    def select_normals(self, key: int | str) -> Any:
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
        out = ""

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
            else:
                out = "\x1b[2m\x1b[1m\x1b[7mMeta Data is empty.\x1b[0m"

        return out


###############################################################################
