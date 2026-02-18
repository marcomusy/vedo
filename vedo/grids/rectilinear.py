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

class RectilinearGrid(PointAlgorithms, MeshVisual):
    """
    Build a rectilinear grid.
    """

    def __init__(self, inputobj=None):
        """
        A RectilinearGrid is a dataset where edges are parallel to the coordinate axes.
        It can be thought of as a tessellation of a box in 3D space, similar to a `Volume`
        except that the cells are not necessarily cubes, but they can have different lengths
        along each axis.
        This can be useful to describe a volume with variable resolution where one needs
        to represent a region with higher detail with respect to another region.

        Arguments:
            inputobj : (vtkRectilinearGrid, list, str)
                list of points and indices, or filename

        Example:
            ```python
            from vedo import RectilinearGrid, show

            xcoords = 7 + np.sqrt(np.arange(0,2500,25))
            ycoords = np.arange(0, 20)
            zcoords = np.arange(0, 20)

            rgrid = RectilinearGrid([xcoords, ycoords, zcoords])

            print(rgrid)
            print(rgrid.x_coordinates().shape)
            print(rgrid.compute_structured_coords([20,10,11]))

            msh = rgrid.tomesh().lw(1)

            show(msh, axes=1, viewup="z")
            ```
        """

        super().__init__()

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

        self.name = "RectilinearGrid"
        self.filename = ""

        self.info = {}
        self.time =  time.time()

        ###############################
        if inputobj is None:
            self.dataset = vtki.vtkRectilinearGrid()

        elif isinstance(inputobj, vtki.vtkRectilinearGrid):
            self.dataset = inputobj

        elif isinstance(inputobj, RectilinearGrid):
            self.dataset = inputobj.dataset

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            if inputobj.endswith(".vtr"):
                reader = vtki.new("XMLRectilinearGridReader")
            else:
                reader = vtki.new("RectilinearGridReader")
            self.filename = inputobj
            reader.SetFileName(inputobj)
            reader.Update()
            self.dataset = reader.GetOutput()

        elif utils.is_sequence(inputobj):
            self.dataset = vtki.vtkRectilinearGrid()
            xcoords, ycoords, zcoords = inputobj
            nx, ny, nz = len(xcoords), len(ycoords), len(zcoords)
            self.dataset.SetDimensions(nx, ny, nz)
            self.dataset.SetXCoordinates(utils.numpy2vtk(xcoords))
            self.dataset.SetYCoordinates(utils.numpy2vtk(ycoords))
            self.dataset.SetZCoordinates(utils.numpy2vtk(zcoords))

        ###############################

        if not self.dataset:
            vedo.logger.error(f"RectilinearGrid: cannot understand input type {type(inputobj)}")
            return

        self.properties.SetColor(0.352, 0.612, 0.996)  # blue7

        self.pipeline = utils.OperationNode(
            self, comment=f"#tets {self.dataset.GetNumberOfCells()}", c="#9e2a2b"
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

    ##################################################################
    def __str__(self):
        """Print a summary for the `RectilinearGrid` object."""
        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(self.memory_address())})".ljust(75),
            c="c", bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m\x1b[36;1m"

        out += "name".ljust(14) + ": " + str(self.name) + "\n"
        if self.filename:
            out += "filename".ljust(14) + ": " + str(self.filename) + "\n"

        out += "dimensions".ljust(14) + ": " + str(self.dataset.GetDimensions()) + "\n"

        out += "center".ljust(14) + ": "
        out += utils.precision(self.dataset.GetCenter(), 6) + "\n"

        bnds = self.bounds()
        bx1, bx2 = utils.precision(bnds[0], 3), utils.precision(bnds[1], 3)
        by1, by2 = utils.precision(bnds[2], 3), utils.precision(bnds[3], 3)
        bz1, bz2 = utils.precision(bnds[4], 3), utils.precision(bnds[5], 3)
        out += "bounds".ljust(14) + ":"
        out += " x=(" + bx1 + ", " + bx2 + "),"
        out += " y=(" + by1 + ", " + by2 + "),"
        out += " z=(" + bz1 + ", " + bz2 + ")\n"

        out += "memory size".ljust(14) + ": "
        out += utils.precision(self.dataset.GetActualMemorySize() / 1024, 2) + " MB\n"

        for key in self.pointdata.keys():
            arr = self.pointdata[key]
            rng = utils.precision(arr.min(), 3) + ", " + utils.precision(arr.max(), 3)
            mark_active = "pointdata"
            a_scalars = self.dataset.GetPointData().GetScalars()
            a_vectors = self.dataset.GetPointData().GetVectors()
            a_tensors = self.dataset.GetPointData().GetTensors()
            if a_scalars and a_scalars.GetName() == key:
                mark_active += " *"
            elif a_vectors and a_vectors.GetName() == key:
                mark_active += " **"
            elif a_tensors and a_tensors.GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), ndim={arr.ndim}'
            out += f", range=({rng})\n"

        for key in self.celldata.keys():
            arr = self.celldata[key]
            rng = utils.precision(arr.min(), 3) + ", " + utils.precision(arr.max(), 3)
            mark_active = "celldata"
            a_scalars = self.dataset.GetCellData().GetScalars()
            a_vectors = self.dataset.GetCellData().GetVectors()
            a_tensors = self.dataset.GetCellData().GetTensors()
            if a_scalars and a_scalars.GetName() == key:
                mark_active += " *"
            elif a_vectors and a_vectors.GetName() == key:
                mark_active += " **"
            elif a_tensors and a_tensors.GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), ndim={arr.ndim}'
            out += f", range=({rng})\n"

        for key in self.metadata.keys():
            arr = self.metadata[key]
            out += "metadata".ljust(14) + ": " + f'"{key}" ({len(arr)} values)\n'

        return out.rstrip() + "\x1b[0m"

    def _repr_html_(self):
        """
        HTML representation of the RectilinearGrid object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.grids.RectilinearGrid"
        help_url = "https://vedo.embl.es/docs/vedo/grids.html#RectilinearGrid"

        m = self.tomesh().linewidth(1).lighting("off")
        arr= m.thumbnail(zoom=1, elevation=-30, azimuth=-30)

        im = Image.fromarray(arr)
        buffered = io.BytesIO()
        im.save(buffered, format="PNG", quality=100)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = "data:image/png;base64," + encoded
        image = f"<img src='{url}'></img>"

        bounds = "<br/>".join(
            [
                utils.precision(min_x, 4) + " ... " + utils.precision(max_x, 4)
                for min_x, max_x in zip(self.bounds()[::2], self.bounds()[1::2])
            ]
        )

        help_text = ""
        if self.name:
            help_text += f"<b> {self.name}: &nbsp&nbsp</b>"
        help_text += '<b><a href="' + help_url + '" target="_blank">' + library_name + "</a></b>"
        if self.filename:
            dots = ""
            if len(self.filename) > 30:
                dots = "..."
            help_text += f"<br/><code><i>({dots}{self.filename[-30:]})</i></code>"

        pdata = ""
        if self.dataset.GetPointData().GetScalars():
            if self.dataset.GetPointData().GetScalars().GetName():
                name = self.dataset.GetPointData().GetScalars().GetName()
                pdata = "<tr><td><b> point data array </b></td><td>" + name + "</td></tr>"

        cdata = ""
        if self.dataset.GetCellData().GetScalars():
            if self.dataset.GetCellData().GetScalars().GetName():
                name = self.dataset.GetCellData().GetScalars().GetName()
                cdata = "<tr><td><b> cell data array </b></td><td>" + name + "</td></tr>"

        pts = self.coordinates
        cm = np.mean(pts, axis=0)

        _all = [
            "<table>",
            "<tr>",
            "<td>", image, "</td>",
            "<td style='text-align: center; vertical-align: center;'><br/>", help_text,
            "<table>",
            "<tr><td><b> bounds </b> <br/> (x/y/z) </td><td>" + str(bounds) + "</td></tr>",
            "<tr><td><b> center of mass </b></td><td>" + utils.precision(cm,3) + "</td></tr>",
            "<tr><td><b> nr. points&nbsp/&nbspcells </b></td><td>"
            + str(self.npoints) + "&nbsp/&nbsp" + str(self.ncells) + "</td></tr>",
            pdata,
            cdata,
            "</table>",
            "</table>",
        ]
        return "\n".join(_all)

    def dimensions(self) -> np.ndarray:
        """Return the number of points in the x, y and z directions."""
        return np.array(self.dataset.GetDimensions())

    def x_coordinates(self) -> np.ndarray:
        """Return the x-coordinates of the grid."""
        return utils.vtk2numpy(self.dataset.GetXCoordinates())

    def y_coordinates(self) -> np.ndarray:
        """Return the y-coordinates of the grid."""
        return utils.vtk2numpy(self.dataset.GetYCoordinates())

    def z_coordinates(self) -> np.ndarray:
        """Return the z-coordinates of the grid."""
        return utils.vtk2numpy(self.dataset.GetZCoordinates())

    def is_point_visible(self, pid: int) -> bool:
        """Return True if point `pid` is visible."""
        return self.dataset.IsPointVisible(pid)

    def is_cell_visible(self, cid: int) -> bool:
        """Return True if cell `cid` is visible."""
        return self.dataset.IsCellVisible(cid)

    def has_blank_points(self) -> bool:
        """Return True if the grid has blank points."""
        return self.dataset.HasAnyBlankPoints()

    def has_blank_cells(self) -> bool:
        """Return True if the grid has blank cells."""
        return self.dataset.HasAnyBlankCells()

    def compute_structured_coords(self, x: list) -> dict:
        """
        Convenience function computes the structured coordinates for a point `x`.

        This method returns a dictionary with keys `ijk`, `pcoords` and `inside`.
        The cell is specified by the array `ijk`.
        and the parametric coordinates in the cell are specified with `pcoords`.
        Value of `inside` is False if the point x is outside of the grid.
        """
        ijk = [0, 0, 0]
        pcoords = [0., 0., 0.]
        inout = self.dataset.ComputeStructuredCoordinates(x, ijk, pcoords)
        return {"ijk": np.array(ijk), "pcoords": np.array(pcoords), "inside": bool(inout)}

    def compute_pointid(self, ijk: int) -> int:
        """Given a location in structured coordinates (i-j-k), return the point id."""
        return self.dataset.ComputePointId(ijk)

    def compute_cellid(self, ijk: int) -> int:
        """Given a location in structured coordinates (i-j-k), return the cell id."""
        return self.dataset.ComputeCellId(ijk)

    def find_point(self, x: list) -> int:
        """Given a position `x`, return the id of the closest point."""
        return self.dataset.FindPoint(x)

    def find_cell(self, x: list) -> dict:
        """Given a position `x`, return the id of the closest cell."""
        cell = vtki.vtkHexagonalPrism()
        cellid = vtki.mutable(0)
        tol2 = 0.001 # vtki.mutable(0)
        subid = vtki.mutable(0)
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0]
        res = self.dataset.FindCell(x, cell, cellid, tol2, subid, pcoords, weights)
        result = {}
        result["cellid"] = cellid
        result["subid"] = subid
        result["pcoords"] = pcoords
        result["weights"] = weights
        result["status"] = res
        return result

    def clone(self, deep=True) -> RectilinearGrid:
        """Return a clone copy of the RectilinearGrid. Alias of `copy()`."""
        if deep:
            newrg = vtki.vtkRectilinearGrid()
            newrg.CopyStructure(self.dataset)
            newrg.CopyAttributes(self.dataset)
            newvol = RectilinearGrid(newrg)
        else:
            newvol = RectilinearGrid(self.dataset)

        prop = vtki.vtkProperty()
        prop.DeepCopy(self.properties)
        newvol.actor.SetProperty(prop)
        newvol.properties = prop
        newvol.pipeline = utils.OperationNode("clone", parents=[self], c="#bbd0ff", shape="diamond")
        return newvol

    def bounds(self) -> np.ndarray:
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        # OVERRIDE CommonAlgorithms.bounds() which is too slow
        return np.array(self.dataset.GetBounds())

    def isosurface(self, value=None) -> vedo.Mesh:
        """
        Return a `Mesh` isosurface extracted from the object.

        Set `value` as single float or list of values to draw the isosurface(s).
        """
        scrange = self.dataset.GetScalarRange()

        cf = vtki.new("ContourFilter")
        cf.UseScalarTreeOn()
        cf.SetInputData(self.dataset)
        cf.ComputeNormalsOn()

        if value is None:
            value = (2 * scrange[0] + scrange[1]) / 3.0
            # print("automatic isosurface value =", value)
            cf.SetValue(0, value)
        else:
            if utils.is_sequence(value):
                cf.SetNumberOfContours(len(value))
                for i, t in enumerate(value):
                    cf.SetValue(i, t)
            else:
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

    def cut_with_plane(self, origin=(0, 0, 0), normal="x") -> UnstructuredGrid:
        """
        Cut the object with the plane defined by a point and a normal.

        Arguments:
            origin : (list)
                the cutting plane goes through this point
            normal : (list, str)
                normal vector to the cutting plane
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
        if isinstance(self, UnstructuredGrid):
            self._update(cout)
            self.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
            return self
        ug.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
        return ug

    def cut_with_mesh(self, mesh, invert=False, whole_cells=False, on_boundary=False) -> UnstructuredGrid:
        """
        Cut a `RectilinearGrid` with a `Mesh`.

        Use `invert` to return cut off part of the input object.
        """
        ug = self.dataset

        ippd = vtki.new("ImplicitPolyDataDistance")
        ippd.SetInput(mesh.dataset)

        if whole_cells or on_boundary:
            clipper = vtki.new("ExtractGeometry")
            clipper.SetInputData(ug)
            clipper.SetImplicitFunction(ippd)
            clipper.SetExtractInside(not invert)
            clipper.SetExtractBoundaryCells(False)
            if on_boundary:
                clipper.SetExtractBoundaryCells(True)
                clipper.SetExtractOnlyBoundaryCells(True)
        else:
            signed_dists = vtki.vtkFloatArray()
            signed_dists.SetNumberOfComponents(1)
            signed_dists.SetName("SignedDistance")
            for pointId in range(ug.GetNumberOfPoints()):
                p = ug.GetPoint(pointId)
                signed_dist = ippd.EvaluateFunction(p)
                signed_dists.InsertNextValue(signed_dist)
            ug.GetPointData().AddArray(signed_dists)
            ug.GetPointData().SetActiveScalars("SignedDistance")  # NEEDED
            clipper = vtki.new("ClipDataSet")
            clipper.SetInputData(ug)
            clipper.SetInsideOut(not invert)
            clipper.SetValue(0.0)

        clipper.Update()

        out = UnstructuredGrid(clipper.GetOutput())
        out.pipeline = utils.OperationNode("cut_with_mesh", parents=[self], c="#9e2a2b")
        return out


