#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from vedo.transformations import LinearTransform

__docformat__ = "google"

__doc__ = """
Work with tetrahedral meshes.

![](https://vedo.embl.es/images/volumetric/82767107-2631d500-9e25-11ea-967c-42558f98f721.jpg)
"""

__all__ = [
    "UnstructuredGrid",
    "TetMesh",
    "RectilinearGrid",
    "StructuredGrid",
]


#########################################################################
class UnstructuredGrid(PointAlgorithms, MeshVisual):
    """Support for UnstructuredGrid objects."""

    def __init__(self, inputobj=None):
        """
        Support for UnstructuredGrid objects.

        Arguments:
            inputobj : (list, vtkUnstructuredGrid, str)
                A list in the form `[points, cells, celltypes]`,
                or a vtkUnstructuredGrid object, or a filename

        Celltypes are identified by the following
        [convention](https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html).
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

        self.name = "UnstructuredGrid"
        self.filename = ""
        self.file_size = ""

        self.info = {}
        self.time = time.time()
        self.rendered_at = set()

        ######################################
        inputtype = str(type(inputobj))

        if inputobj is None:
            self.dataset = vtki.vtkUnstructuredGrid()

        elif utils.is_sequence(inputobj):

            pts, cells, celltypes = inputobj
            assert len(cells) == len(celltypes)

            self.dataset = vtki.vtkUnstructuredGrid()

            if not utils.is_sequence(cells[0]):
                tets = []
                nf = cells[0] + 1
                for i, cl in enumerate(cells):
                    if i in (nf, 0):
                        k = i + 1
                        nf = cl + k
                        cell = [cells[j + k] for j in range(cl)]
                        tets.append(cell)
                cells = tets

            # This would fill the points and use those to define orientation
            vpts = utils.numpy2vtk(pts, dtype=np.float32)
            points = vtki.vtkPoints()
            points.SetData(vpts)
            self.dataset.SetPoints(points)

            # Fill cells
            # https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
            for i, ct in enumerate(celltypes):
                if   ct == vtki.cell_types["VERTEX"]:
                    cell = vtki.vtkVertex()
                elif ct == vtki.cell_types["POLY_VERTEX"]:
                    cell = vtki.vtkPolyVertex()
                elif ct == vtki.cell_types["TETRA"]:
                    cell = vtki.vtkTetra()
                elif ct == vtki.cell_types["WEDGE"]:
                    cell = vtki.vtkWedge()
                elif ct == vtki.cell_types["LINE"]:
                    cell = vtki.vtkLine()
                elif ct == vtki.cell_types["POLY_LINE"]:
                    cell = vtki.vtkPolyLine()
                elif ct == vtki.cell_types["TRIANGLE"]:
                    cell = vtki.vtkTriangle()
                elif ct == vtki.cell_types["TRIANGLE_STRIP"]:
                    cell = vtki.vtkTriangleStrip()
                elif ct == vtki.cell_types["POLYGON"]:
                    cell = vtki.vtkPolygon()
                elif ct == vtki.cell_types["PIXEL"]:
                    cell = vtki.vtkPixel()
                elif ct == vtki.cell_types["QUAD"]:
                    cell = vtki.vtkQuad()
                elif ct == vtki.cell_types["VOXEL"]:
                    cell = vtki.vtkVoxel()
                elif ct == vtki.cell_types["PYRAMID"]:
                    cell = vtki.vtkPyramid()
                elif ct == vtki.cell_types["HEXAHEDRON"]:
                    cell = vtki.vtkHexahedron()
                elif ct == vtki.cell_types["HEXAGONAL_PRISM"]:
                    cell = vtki.vtkHexagonalPrism()
                elif ct == vtki.cell_types["PENTAGONAL_PRISM"]:
                    cell = vtki.vtkPentagonalPrism()
                elif ct == vtki.cell_types["QUADRATIC_TETRA"]:
                    from vtkmodules.vtkCommonDataModel import vtkQuadraticTetra
                    cell = vtkQuadraticTetra()
                elif ct == vtki.cell_types["QUADRATIC_HEXAHEDRON"]:
                    from vtkmodules.vtkCommonDataModel import vtkQuadraticHexahedron
                    cell = vtkQuadraticHexahedron()
                elif ct == vtki.cell_types["QUADRATIC_WEDGE"]:
                    from vtkmodules.vtkCommonDataModel import vtkQuadraticWedge
                    cell = vtkQuadraticWedge()
                elif ct == vtki.cell_types["QUADRATIC_PYRAMID"]:
                    from vtkmodules.vtkCommonDataModel import vtkQuadraticPyramid
                    cell = vtkQuadraticPyramid()
                elif ct == vtki.cell_types["QUADRATIC_LINEAR_QUAD"]:
                    from vtkmodules.vtkCommonDataModel import vtkQuadraticLinearQuad
                    cell = vtkQuadraticLinearQuad()
                elif ct == vtki.cell_types["QUADRATIC_LINEAR_WEDGE"]:
                    from vtkmodules.vtkCommonDataModel import vtkQuadraticLinearWedge
                    cell = vtkQuadraticLinearWedge()
                elif ct == vtki.cell_types["BIQUADRATIC_QUADRATIC_WEDGE"]:
                    from vtkmodules.vtkCommonDataModel import vtkBiQuadraticQuadraticWedge
                    cell = vtkBiQuadraticQuadraticWedge()
                elif ct == vtki.cell_types["BIQUADRATIC_QUADRATIC_HEXAHEDRON"]:
                    from vtkmodules.vtkCommonDataModel import vtkBiQuadraticQuadraticHexahedron
                    cell = vtkBiQuadraticQuadraticHexahedron()
                elif ct == vtki.cell_types["BIQUADRATIC_TRIANGLE"]:
                    from vtkmodules.vtkCommonDataModel import vtkBiQuadraticTriangle
                    cell = vtkBiQuadraticTriangle()
                elif ct == vtki.cell_types["CUBIC_LINE"]:
                    from vtkmodules.vtkCommonDataModel import vtkCubicLine
                    cell = vtkCubicLine()
                elif ct == vtki.cell_types["CONVEX_POINT_SET"]:
                    from vtkmodules.vtkCommonDataModel import vtkConvexPointSet
                    cell = vtkConvexPointSet()
                elif ct == vtki.cell_types["POLYHEDRON"]:
                    from vtkmodules.vtkCommonDataModel import vtkPolyhedron
                    cell = vtkPolyhedron()
                elif ct == vtki.cell_types["HIGHER_ORDER_TRIANGLE"]:
                    from vtkmodules.vtkCommonDataModel import vtkHigherOrderTriangle
                    cell = vtkHigherOrderTriangle()
                elif ct == vtki.cell_types["HIGHER_ORDER_QUAD"]:
                    from vtkmodules.vtkCommonDataModel import vtkHigherOrderQuadrilateral
                    cell = vtkHigherOrderQuadrilateral()
                elif ct == vtki.cell_types["HIGHER_ORDER_TETRAHEDRON"]:
                    from vtkmodules.vtkCommonDataModel import vtkHigherOrderTetra
                    cell = vtkHigherOrderTetra()
                elif ct == vtki.cell_types["HIGHER_ORDER_WEDGE"]:
                    from vtkmodules.vtkCommonDataModel import vtkHigherOrderWedge
                    cell = vtkHigherOrderWedge()
                elif ct == vtki.cell_types["HIGHER_ORDER_HEXAHEDRON"]:
                    from vtkmodules.vtkCommonDataModel import vtkHigherOrderHexahedron
                    cell = vtkHigherOrderHexahedron()
                elif ct == vtki.cell_types["LAGRANGE_CURVE"]:
                    from vtkmodules.vtkCommonDataModel import vtkLagrangeCurve
                    cell = vtkLagrangeCurve()
                elif ct == vtki.cell_types["LAGRANGE_TRIANGLE"]:
                    from vtkmodules.vtkCommonDataModel import vtkLagrangeTriangle
                    cell = vtkLagrangeTriangle()
                elif ct == vtki.cell_types["LAGRANGE_QUADRILATERAL"]:
                    from vtkmodules.vtkCommonDataModel import vtkLagrangeQuadrilateral
                    cell = vtkLagrangeQuadrilateral()
                elif ct == vtki.cell_types["LAGRANGE_TETRAHEDRON"]:
                    from vtkmodules.vtkCommonDataModel import vtkLagrangeTetra
                    cell = vtkLagrangeTetra()
                elif ct == vtki.cell_types["LAGRANGE_HEXAHEDRON"]:
                    from vtkmodules.vtkCommonDataModel import vtkLagrangeHexahedron
                    cell = vtkLagrangeHexahedron()
                elif ct == vtki.cell_types["LAGRANGE_WEDGE"]:
                    from vtkmodules.vtkCommonDataModel import vtkLagrangeWedge
                    cell = vtkLagrangeWedge()
                elif ct == vtki.cell_types["BEZIER_CURVE"]:
                    from vtkmodules.vtkCommonDataModel import vtkBezierCurve
                    cell = vtkBezierCurve()
                elif ct == vtki.cell_types["BEZIER_TRIANGLE"]:
                    from vtkmodules.vtkCommonDataModel import vtkBezierTriangle
                    cell = vtkBezierTriangle()
                elif ct == vtki.cell_types["BEZIER_QUADRILATERAL"]:
                    from vtkmodules.vtkCommonDataModel import vtkBezierQuadrilateral
                    cell = vtkBezierQuadrilateral()
                elif ct == vtki.cell_types["BEZIER_TETRAHEDRON"]:
                    from vtkmodules.vtkCommonDataModel import vtkBezierTetra
                    cell = vtkBezierTetra()
                elif ct == vtki.cell_types["BEZIER_HEXAHEDRON"]:
                    from vtkmodules.vtkCommonDataModel import vtkBezierHexahedron
                    cell = vtkBezierHexahedron()
                elif ct == vtki.cell_types["BEZIER_WEDGE"]:
                    from vtkmodules.vtkCommonDataModel import vtkBezierWedge
                    cell = vtkBezierWedge()
                else:
                    vedo.logger.error(
                        f"UnstructuredGrid: cell type {ct} not supported. Skip.")
                    continue

                cpids = cell.GetPointIds()
                cell_conn = cells[i]
                for j, pid in enumerate(cell_conn):
                    cpids.SetId(j, pid)
                self.dataset.InsertNextCell(ct, cpids)

        elif "UnstructuredGrid" in inputtype:
            self.dataset = inputobj

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            self.filename = inputobj
            if inputobj.endswith(".vtu"):
                reader = vtki.new("XMLUnstructuredGridReader")
            else:
                reader = vtki.new("UnstructuredGridReader")
            self.filename = inputobj
            reader.SetFileName(inputobj)
            reader.Update()
            self.dataset = reader.GetOutput()

        else:
            # this converts other types of vtk objects to UnstructuredGrid
            apf = vtki.new("AppendFilter")
            try:
                apf.AddInputData(inputobj)
            except TypeError:
                apf.AddInputData(inputobj.dataset)
            apf.Update()
            self.dataset = apf.GetOutput()

        self.properties.SetColor(0.89, 0.455, 0.671)  # pink7

        self.pipeline = utils.OperationNode(
            self, comment=f"#cells {self.dataset.GetNumberOfCells()}", c="#4cc9f0"
        )

    # ------------------------------------------------------------------
    def __str__(self):
        """Print a string summary of the `UnstructuredGrid` object."""
        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(self.memory_address())})".ljust(75),
            c="m", bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m\u001b[35m"

        out += "nr. of verts".ljust(14) + ": " + str(self.npoints) + "\n"
        out += "nr. of cells".ljust(14) + ": " + str(self.ncells)  + "\n"
        ct_arr = np.unique(self.cell_types_array)
        cnames = [k for k, v in vtki.cell_types.items() if v in ct_arr]
        out += "cell types".ljust(14) + ": " + str(cnames) + "\n"

        if self.npoints:
            out+="size".ljust(14)+ ": average=" + utils.precision(self.average_size(),6)
            out+=", diagonal="+ utils.precision(self.diagonal_size(), 6)+ "\n"
            out+="center of mass".ljust(14) + ": " + utils.precision(self.center_of_mass(),6)+"\n"

        bnds = self.bounds()
        bx1, bx2 = utils.precision(bnds[0], 3), utils.precision(bnds[1], 3)
        by1, by2 = utils.precision(bnds[2], 3), utils.precision(bnds[3], 3)
        bz1, bz2 = utils.precision(bnds[4], 3), utils.precision(bnds[5], 3)
        out += "bounds".ljust(14) + ":"
        out += " x=(" + bx1 + ", " + bx2 + "),"
        out += " y=(" + by1 + ", " + by2 + "),"
        out += " z=(" + bz1 + ", " + bz2 + ")\n"

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
        HTML representation of the UnstructuredGrid object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.grids.UnstructuredGrid"
        help_url = "https://vedo.embl.es/docs/vedo/grids.html#UnstructuredGrid"

        arr = self.thumbnail()
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
            # "<tr><td><b> average size </b></td><td>" + str(average_size) + "</td></tr>",
            "<tr><td><b> nr. points&nbsp/&nbspcells </b></td><td>"
            + str(self.npoints) + "&nbsp/&nbsp" + str(self.ncells) + "</td></tr>",
            pdata,
            cdata,
            "</table>",
            "</table>",
        ]
        return "\n".join(_all)

    @property
    def actor(self):
        """Return the `vtkActor` of the object."""
        # print("building actor")
        gf = vtki.new("GeometryFilter")
        gf.SetInputData(self.dataset)
        gf.Update()
        out = gf.GetOutput()
        self.mapper.SetInputData(out)
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

    def merge(self, *others) -> Self:
        """
        Merge multiple datasets into one single `UnstrcturedGrid`.
        """
        apf = vtki.new("AppendFilter")
        for o in others:
            if isinstance(o, UnstructuredGrid):
                apf.AddInputData(o.dataset)
            elif isinstance(o, vtki.vtkUnstructuredGrid):
                apf.AddInputData(o)
            else:
                vedo.printc("Error: cannot merge type", type(o), c="r")
        apf.Update()
        self._update(apf.GetOutput())
        self.pipeline = utils.OperationNode(
            "merge", parents=[self, *others], c="#9e2a2b"
        )
        return self

    def copy(self, deep=True) -> "UnstructuredGrid":
        """Return a copy of the object. Alias of `clone()`."""
        return self.clone(deep=deep)

    def clone(self, deep=True) -> "UnstructuredGrid":
        """Clone the UnstructuredGrid object to yield an exact copy."""
        ug = vtki.vtkUnstructuredGrid()
        if deep:
            ug.DeepCopy(self.dataset)
        else:
            ug.ShallowCopy(self.dataset)
        if isinstance(self, vedo.UnstructuredGrid):
            cloned = vedo.UnstructuredGrid(ug)
        else:
            cloned = vedo.TetMesh(ug)

        cloned.copy_properties_from(self)

        cloned.pipeline = utils.OperationNode(
            "clone", parents=[self], shape="diamond", c="#bbe1ed"
        )
        return cloned

    def bounds(self) -> np.ndarray:
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        # OVERRIDE CommonAlgorithms.bounds() which is too slow
        return np.array(self.dataset.GetBounds())

    def threshold(self, name=None, above=None, below=None, on="cells") -> Self:
        """
        Threshold the tetrahedral mesh by a cell scalar value.
        Reduce to only cells which satisfy the threshold limits.

        - if `above = below` will only select cells with that specific value.
        - if `above > below` selection range is flipped.

        Set keyword "on" to either "cells" or "points".
        """
        th = vtki.new("Threshold")
        th.SetInputData(self.dataset)

        if name is None:
            if self.celldata.keys():
                name = self.celldata.keys()[0]
                th.SetInputArrayToProcess(0, 0, 0, 1, name)
            elif self.pointdata.keys():
                name = self.pointdata.keys()[0]
                th.SetInputArrayToProcess(0, 0, 0, 0, name)
            if name is None:
                vedo.logger.warning("cannot find active array. Skip.")
                return self
        else:
            if on.startswith("c"):
                th.SetInputArrayToProcess(0, 0, 0, 1, name)
            else:
                th.SetInputArrayToProcess(0, 0, 0, 0, name)

        if above is not None:
            th.SetLowerThreshold(above)

        if below is not None:
            th.SetUpperThreshold(below)

        th.Update()
        return self._update(th.GetOutput())

    def isosurface(self, value=None, flying_edges=False) -> "vedo.mesh.Mesh":
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

        out = vedo.mesh.Mesh(poly, c=None).flat()
        out.mapper.SetScalarRange(scrange[0], scrange[1])

        out.pipeline = utils.OperationNode(
            "isosurface",
            parents=[self],
            comment=f"#pts {out.dataset.GetNumberOfPoints()}",
            c="#4cc9f0:#e9c46a",
        )
        return out

    def shrink(self, fraction=0.8) -> Self:
        """
        Shrink the individual cells.

        ![](https://vedo.embl.es/images/feats/shrink_hex.png)
        """
        sf = vtki.new("ShrinkFilter")
        sf.SetInputData(self.dataset)
        sf.SetShrinkFactor(fraction)
        sf.Update()
        out = sf.GetOutput()
        self._update(out)
        self.pipeline = utils.OperationNode(
            "shrink", comment=f"by {fraction}", parents=[self], c="#9e2a2b"
        )
        return self

    def tomesh(self, fill=False, shrink=1.0) -> "vedo.mesh.Mesh":
        """
        Build a polygonal `Mesh` from the current object.

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
        else:
            gf.SetInputData(self.dataset)
            gf.Update()
            poly = gf.GetOutput()

        msh = vedo.mesh.Mesh(poly)
        msh.copy_properties_from(self)
        msh.pipeline = utils.OperationNode(
            "tomesh", parents=[self], comment=f"fill={fill}", c="#9e2a2b:#e9c46a"
        )
        return msh

    @property
    def cell_types_array(self):
        """Return the list of cell types in the dataset."""
        uarr = self.dataset.GetCellTypesArray()
        return utils.vtk2numpy(uarr)

    def extract_cells_by_type(self, ctype) -> "UnstructuredGrid":
        """Extract a specific cell type and return a new `UnstructuredGrid`."""
        if isinstance(ctype, str):
            try:
                ctype = vtki.cell_types[ctype.upper()]
            except KeyError:
                vedo.logger.error(f"extract_cells_by_type: cell type {ctype} does not exist. Skip.")
                return self
        uarr = self.dataset.GetCellTypesArray()
        ctarrtyp = np.where(utils.vtk2numpy(uarr) == ctype)[0]
        uarrtyp = utils.numpy2vtk(ctarrtyp, deep=False, dtype="id")
        selection_node = vtki.new("SelectionNode")
        selection_node.SetFieldType(vtki.get_class("SelectionNode").CELL)
        selection_node.SetContentType(vtki.get_class("SelectionNode").INDICES)
        selection_node.SetSelectionList(uarrtyp)
        selection = vtki.new("Selection")
        selection.AddNode(selection_node)
        es = vtki.new("ExtractSelection")
        es.SetInputData(0, self.dataset)
        es.SetInputData(1, selection)
        es.Update()

        ug = UnstructuredGrid(es.GetOutput())
        ug.pipeline = utils.OperationNode(
            "extract_cell_type", comment=f"type {ctype}", c="#edabab", parents=[self]
        )
        return ug

    def extract_cells_by_id(self, idlist, use_point_ids=False) -> "UnstructuredGrid":
        """Return a new `UnstructuredGrid` composed of the specified subset of indices."""
        selection_node = vtki.new("SelectionNode")
        if use_point_ids:
            selection_node.SetFieldType(vtki.get_class("SelectionNode").POINT)
            contcells = vtki.get_class("SelectionNode").CONTAINING_CELLS()
            selection_node.GetProperties().Set(contcells, 1)
        else:
            selection_node.SetFieldType(vtki.get_class("SelectionNode").CELL)
        selection_node.SetContentType(vtki.get_class("SelectionNode").INDICES)
        vidlist = utils.numpy2vtk(idlist, dtype="id")
        selection_node.SetSelectionList(vidlist)
        selection = vtki.new("Selection")
        selection.AddNode(selection_node)
        es = vtki.new("ExtractSelection")
        es.SetInputData(0, self)
        es.SetInputData(1, selection)
        es.Update()

        ug = UnstructuredGrid(es.GetOutput())
        pr = vtki.vtkProperty()
        pr.DeepCopy(self.properties)
        ug.actor.SetProperty(pr)
        ug.properties = pr

        ug.mapper.SetLookupTable(utils.ctf2lut(self))
        ug.pipeline = utils.OperationNode(
            "extract_cells_by_id",
            parents=[self],
            comment=f"#cells {self.dataset.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        return ug

    def find_cell(self, p: list) -> int:
        """Locate the cell that contains a point and return the cell ID."""
        if self.cell_locator is None:
            self.cell_locator = vtki.new("CellLocator")
            self.cell_locator.SetDataSet(self.dataset)
            self.cell_locator.BuildLocator()
        cid = self.cell_locator.FindCell(p)
        return cid

    def clean(self) -> Self:
        """
        Cleanup unused points and empty cells
        """
        cl = vtki.new("StaticCleanUnstructuredGrid")
        cl.SetInputData(self.dataset)
        cl.RemoveUnusedPointsOn()
        cl.ProduceMergeMapOff()
        cl.AveragePointDataOff()
        cl.Update()

        self._update(cl.GetOutput())
        self.pipeline = utils.OperationNode(
            "clean",
            parents=[self],
            comment=f"#cells {self.dataset.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        return self

    def extract_cells_on_plane(self, origin: tuple, normal: tuple) -> Self:
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtki.new("3DLinearGridCrinkleExtractor")
        bf.SetInputData(self.dataset)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        plane = vtki.new("Plane")
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        bf.SetImplicitFunction(plane)
        bf.Update()

        self._update(bf.GetOutput(), reset_locators=False)
        self.pipeline = utils.OperationNode(
            "extract_cells_on_plane",
            parents=[self],
            comment=f"#cells {self.dataset.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        return self

    def extract_cells_on_sphere(self, center: tuple, radius: tuple) -> Self:
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtki.new("3DLinearGridCrinkleExtractor")
        bf.SetInputData(self.dataset)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        sph = vtki.new("Sphere")
        sph.SetRadius(radius)
        sph.SetCenter(center)
        bf.SetImplicitFunction(sph)
        bf.Update()

        self._update(bf.GetOutput())
        self.pipeline = utils.OperationNode(
            "extract_cells_on_sphere",
            parents=[self],
            comment=f"#cells {self.dataset.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        return self

    def extract_cells_on_cylinder(self, center: tuple, axis: tuple, radius: float) -> Self:
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtki.new("3DLinearGridCrinkleExtractor")
        bf.SetInputData(self.dataset)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        cyl = vtki.new("Cylinder")
        cyl.SetRadius(radius)
        cyl.SetCenter(center)
        cyl.SetAxis(axis)
        bf.SetImplicitFunction(cyl)
        bf.Update()

        self.pipeline = utils.OperationNode(
            "extract_cells_on_cylinder",
            parents=[self],
            comment=f"#cells {self.dataset.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        self._update(bf.GetOutput())
        return self

    def cut_with_plane(self, origin=(0, 0, 0), normal="x") -> "UnstructuredGrid":
        """
        Cut the object with the plane defined by a point and a normal.

        Arguments:
            origin : (list)
                the cutting plane goes through this point
            normal : (list, str)
                normal vector to the cutting plane
        """
        # if isinstance(self, vedo.Volume):
        #     raise RuntimeError("cut_with_plane() is not applicable to Volume objects.")

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

        if isinstance(cout, vtki.vtkUnstructuredGrid):
            ug = vedo.UnstructuredGrid(cout)
            if isinstance(self, vedo.UnstructuredGrid):
                self._update(cout)
                self.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
                return self
            ug.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
            return ug

        else:
            self._update(cout)
            self.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
            return self

    def cut_with_box(self, box: Any) -> "UnstructuredGrid":
        """
        Cut the grid with the specified bounding box.

        Parameter box has format [xmin, xmax, ymin, ymax, zmin, zmax].
        If an object is passed, its bounding box are used.

        This method always returns a TetMesh object.

        Example:
        ```python
        from vedo import *
        tmesh = TetMesh(dataurl+'limb_ugrid.vtk')
        tmesh.color('rainbow')
        cu = Cube(side=500).x(500) # any Mesh works
        tmesh.cut_with_box(cu).show(axes=1)
        ```

        ![](https://vedo.embl.es/images/feats/tet_cut_box.png)
        """
        bc = vtki.new("BoxClipDataSet")
        bc.SetInputData(self.dataset)
        try:
            boxb = box.bounds()
        except AttributeError:
            boxb = box

        bc.SetBoxClip(*boxb)
        bc.Update()
        cout = bc.GetOutput()

        # output of vtkBoxClipDataSet is always tetrahedrons
        tm = vedo.TetMesh(cout)
        tm.pipeline = utils.OperationNode("cut_with_box", parents=[self], c="#9e2a2b")
        return tm

    def cut_with_mesh(self, mesh: Mesh, invert=False, whole_cells=False, on_boundary=False) -> "UnstructuredGrid":
        """
        Cut a `UnstructuredGrid` or `TetMesh` with a `Mesh`.

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

        out = vedo.UnstructuredGrid(clipper.GetOutput())
        out.pipeline = utils.OperationNode("cut_with_mesh", parents=[self], c="#9e2a2b")
        return out


##########################################################################
class TetMesh(UnstructuredGrid):
    """The class describing tetrahedral meshes."""

    def __init__(self, inputobj=None):
        """
        Arguments:
            inputobj : (vtkUnstructuredGrid, list, str, tetgenpy.TetgenIO)
                list of points and tet indices, or filename
        """
        super().__init__()

        self.dataset = None

        self.mapper = vtki.new("PolyDataMapper")
        self._actor = vtki.vtkActor()
        self._actor.retrieve_object = weak_ref_to(self)
        self._actor.SetMapper(self.mapper)
        self.properties = self._actor.GetProperty()

        self.name = "TetMesh"

        # print('TetMesh inputtype', type(inputobj))

        ###################
        if inputobj is None:
            self.dataset = vtki.vtkUnstructuredGrid()

        elif isinstance(inputobj, vtki.vtkUnstructuredGrid):
            self.dataset = inputobj

        elif isinstance(inputobj, UnstructuredGrid):
            self.dataset = inputobj.dataset

        elif "TetgenIO" in str(type(inputobj)):  # tetgenpy object
            inputobj = [inputobj.points(), inputobj.tetrahedra()]

        elif isinstance(inputobj, vtki.vtkRectilinearGrid):
            r2t = vtki.new("RectilinearGridToTetrahedra")
            r2t.SetInputData(inputobj)
            r2t.RememberVoxelIdOn()
            r2t.SetTetraPerCellTo6()
            r2t.Update()
            self.dataset = r2t.GetOutput()

        elif isinstance(inputobj, vtki.vtkDataSet):
            r2t = vtki.new("DataSetTriangleFilter")
            r2t.SetInputData(inputobj)
            r2t.TetrahedraOnlyOn()
            r2t.Update()
            self.dataset = r2t.GetOutput()

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            if inputobj.endswith(".vtu"):
                reader = vtki.new("XMLUnstructuredGridReader")
            else:
                reader = vtki.new("UnstructuredGridReader")

            if not os.path.isfile(inputobj):
                # for some reason vtk Reader does not complain
                vedo.logger.error(f"file {inputobj} not found")
                raise FileNotFoundError

            self.filename = inputobj
            reader.SetFileName(inputobj)
            reader.Update()
            ug = reader.GetOutput()

            tt = vtki.new("DataSetTriangleFilter")
            tt.SetInputData(ug)
            tt.SetTetrahedraOnly(True)
            tt.Update()
            self.dataset = tt.GetOutput()

        ###############################
        if utils.is_sequence(inputobj):
            self.dataset = vtki.vtkUnstructuredGrid()

            points, cells = inputobj
            if len(points) == 0:
                return
            if not utils.is_sequence(points[0]):
                return
            if len(cells) == 0:
                return

            if not utils.is_sequence(cells[0]):
                tets = []
                nf = cells[0] + 1
                for i, cl in enumerate(cells):
                    if i in (nf, 0):
                        k = i + 1
                        nf = cl + k
                        cell = [cells[j + k] for j in range(cl)]
                        tets.append(cell)
                cells = tets

            source_points = vtki.vtkPoints()
            varr = utils.numpy2vtk(points, dtype=np.float32)
            source_points.SetData(varr)
            self.dataset.SetPoints(source_points)

            source_tets = vtki.vtkCellArray()
            for f in cells:
                ele = vtki.vtkTetra()
                pid = ele.GetPointIds()
                for i, fi in enumerate(f):
                    pid.SetId(i, fi)
                source_tets.InsertNextCell(ele)
            self.dataset.SetCells(vtki.cell_types["TETRA"], source_tets)

        if not self.dataset:
            vedo.logger.error(f"cannot understand input type {type(inputobj)}")
            return

        self.properties.SetColor(0.352, 0.612, 0.996)  # blue7

        self.pipeline = utils.OperationNode(
            self, comment=f"#tets {self.dataset.GetNumberOfCells()}", c="#9e2a2b"
        )

    ##################################################################
    def __str__(self):
        """Print a string summary of the `TetMesh` object."""
        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(self.memory_address())})".ljust(75),
            c="c", bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m\u001b[36m"

        out += "nr. of verts".ljust(14) + ": " + str(self.npoints) + "\n"
        out += "nr. of tetras".ljust(14) + ": " + str(self.ncells) + "\n"

        if self.npoints:
            out+="size".ljust(14)+ ": average=" + utils.precision(self.average_size(),6)
            out+=", diagonal="+ utils.precision(self.diagonal_size(), 6)+ "\n"
            out+="center of mass".ljust(14) + ": " + utils.precision(self.center_of_mass(),6)+"\n"

        bnds = self.bounds()
        bx1, bx2 = utils.precision(bnds[0], 3), utils.precision(bnds[1], 3)
        by1, by2 = utils.precision(bnds[2], 3), utils.precision(bnds[3], 3)
        bz1, bz2 = utils.precision(bnds[4], 3), utils.precision(bnds[5], 3)
        out += "bounds".ljust(14) + ":"
        out += " x=(" + bx1 + ", " + bx2 + "),"
        out += " y=(" + by1 + ", " + by2 + "),"
        out += " z=(" + bz1 + ", " + bz2 + ")\n"

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
        HTML representation of the TetMesh object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.grids.TetMesh"
        help_url = "https://vedo.embl.es/docs/vedo/grids.html#TetMesh"

        arr = self.thumbnail()
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

        allt = [
            "<table>",
            "<tr>",
            "<td>", image, "</td>",
            "<td style='text-align: center; vertical-align: center;'><br/>", help_text,
            "<table>",
            "<tr><td><b> bounds </b> <br/> (x/y/z) </td><td>" + str(bounds) + "</td></tr>",
            "<tr><td><b> center of mass </b></td><td>" + utils.precision(cm,3) + "</td></tr>",
            "<tr><td><b> nr. points&nbsp/&nbsptets </b></td><td>"
            + str(self.npoints) + "&nbsp/&nbsp" + str(self.ncells) + "</td></tr>",
            pdata,
            cdata,
            "</table>",
            "</table>",
        ]
        return "\n".join(allt)

    def compute_quality(self, metric=7) -> np.ndarray:
        """
        Calculate functions of quality for the elements of a tetrahedral mesh.
        This method adds to the mesh a cell array named "Quality".

        Arguments:
            metric : (int)
                type of estimators:
                    - EDGE RATIO, 0
                    - ASPECT RATIO, 1
                    - RADIUS RATIO, 2
                    - ASPECT FROBENIUS, 3
                    - MIN_ANGLE, 4
                    - COLLAPSE RATIO, 5
                    - ASPECT GAMMA, 6
                    - VOLUME, 7
                    - ...

        See class [vtkMeshQuality](https://vtk.org/doc/nightly/html/classvtkMeshQuality.html)
        for an explanation of the meaning of each metric..
        """
        qf = vtki.new("MeshQuality")
        qf.SetInputData(self.dataset)
        qf.SetTetQualityMeasure(metric)
        qf.SaveCellQualityOn()
        qf.Update()
        self._update(qf.GetOutput())
        return utils.vtk2numpy(qf.GetOutput().GetCellData().GetArray("Quality"))

    def check_validity(self, tol=0) -> np.ndarray:
        """
        Return an array of possible problematic tets following this convention:
        ```python
        Valid               =  0
        WrongNumberOfPoints = 01
        IntersectingEdges   = 02
        IntersectingFaces   = 04
        NoncontiguousEdges  = 08
        Nonconvex           = 10
        OrientedIncorrectly = 20
        ```

        Arguments:
            tol : (float)
                This value is used as an epsilon for floating point
                equality checks throughout the cell checking process.
        """
        vald = vtki.new("CellValidator")
        if tol:
            vald.SetTolerance(tol)
        vald.SetInputData(self.dataset)
        vald.Update()
        varr = vald.GetOutput().GetCellData().GetArray("ValidityState")
        return utils.vtk2numpy(varr)

    def decimate(self, scalars_name: str, fraction=0.5, n=0) -> Self:
        """
        Downsample the number of tets in a TetMesh to a specified fraction.
        Either `fraction` or `n` must be set.

        Arguments:
            fraction : (float)
                the desired final fraction of the total.
            n : (int)
                the desired number of final tets

        .. note:: setting `fraction=0.1` leaves 10% of the original nr of tets.
        """
        decimate = vtki.new("UnstructuredGridQuadricDecimation")
        decimate.SetInputData(self.dataset)
        decimate.SetScalarsName(scalars_name)

        if n:  # n = desired number of points
            decimate.SetNumberOfTetsOutput(n)
        else:
            decimate.SetTargetReduction(1 - fraction)
        decimate.Update()
        self._update(decimate.GetOutput())
        self.pipeline = utils.OperationNode(
            "decimate", comment=f"array: {scalars_name}", c="#edabab", parents=[self]
        )
        return self

    def subdivide(self) -> Self:
        """
        Increase the number of tetrahedrons of a `TetMesh`.
        Subdivides each tetrahedron into twelve smaller tetras.
        """
        sd = vtki.new("SubdivideTetra")
        sd.SetInputData(self.dataset)
        sd.Update()
        self._update(sd.GetOutput())
        self.pipeline = utils.OperationNode("subdivide", c="#edabab", parents=[self])
        return self

    def generate_random_points(self, n, min_radius=0) -> "vedo.Points":
        """
        Generate `n` uniformly distributed random points
        inside the tetrahedral mesh.

        A new point data array is added to the output points
        called "OriginalCellID" which contains the index of
        the cell ID in which the point was generated.

        Arguments:
            n : (int)
                number of points to generate.
            min_radius: (float)
                impose a minimum distance between points.
                If `min_radius` is set to 0, the points are
                generated uniformly at random inside the mesh.
                If `min_radius` is set to a positive value,
                the points are generated uniformly at random
                inside the mesh, but points closer than `min_radius`
                to any other point are discarded.

        Returns a `vedo.Points` object.

        Note:
            Consider using `points.probe(msh)` to interpolate
            any existing mesh data onto the points.

        Example:
        ```python
        from vedo import *
        tmesh = TetMesh(dataurl + "limb.vtu").alpha(0.2)
        pts = tmesh.generate_random_points(20000, min_radius=10)
        print(pts.pointdata["OriginalCellID"])
        show(pts, tmesh, axes=1).close()
        ```
        """
        cmesh = self.compute_cell_size()
        tets = cmesh.cells
        verts = cmesh.coordinates
        cumul = np.cumsum(np.abs(cmesh.celldata["Volume"]))

        out_pts = []
        orig_cell = []
        for _ in range(n):
            random_area = np.random.random() * cumul[-1]
            it = np.searchsorted(cumul, random_area)
            A, B, C, D = verts[tets[it]]
            r1, r2, r3 = sorted(np.random.random(3))
            p = r1 * A + (r2 - r1) * B + (r3 - r2) * C + (1 - r3) * D
            out_pts.append(p)
            orig_cell.append(it)
        orig_cellnp = np.array(orig_cell, dtype=np.uint32)

        vpts = vedo.pointcloud.Points(out_pts)
        vpts.pointdata["OriginalCellID"] = orig_cellnp

        if min_radius > 0:
            vpts.subsample(min_radius, absolute=True)

        vpts.point_size(5).color("k1")
        vpts.name = "RandomPoints"
        vpts.pipeline = utils.OperationNode(
            "generate_random_points", c="#edabab", parents=[self])
        return vpts

    def isosurface(self, value=True, flying_edges=None) -> "vedo.Mesh":
        """
        Return a `vedo.Mesh` isosurface.
        The "isosurface" is the surface of the region of points
        whose values equal to `value`.

        Set `value` to a single value or list of values to compute the isosurface(s).

        Note that flying_edges option is not available for `TetMesh`.
        """
        if flying_edges is not None:
            vedo.logger.warning("flying_edges option is not available for TetMesh.")

        if not self.dataset.GetPointData().GetScalars():
            vedo.logger.warning(
                "in isosurface() no scalar pointdata found. "
                "Mappping cells to points."
            )
            self.map_cells_to_points()
        scrange = self.dataset.GetPointData().GetScalars().GetRange()
        cf = vtki.new("ContourFilter")  # vtki.new("ContourGrid")
        cf.SetInputData(self.dataset)

        if utils.is_sequence(value):
            cf.SetNumberOfContours(len(value))
            for i, t in enumerate(value):
                cf.SetValue(i, t)
            cf.Update()
        else:
            if value is True:
                value = (2 * scrange[0] + scrange[1]) / 3.0
            cf.SetValue(0, value)
            cf.Update()

        msh = Mesh(cf.GetOutput(), c=None)
        msh.copy_properties_from(self)
        msh.pipeline = utils.OperationNode("isosurface", c="#edabab", parents=[self])
        return msh

    def slice(self, origin=(0, 0, 0), normal=(1, 0, 0)) -> "vedo.Mesh":
        """
        Return a 2D slice of the mesh by a plane passing through origin and assigned normal.
        """
        strn = str(normal)
        if   strn ==  "x": normal = (1, 0, 0)
        elif strn ==  "y": normal = (0, 1, 0)
        elif strn ==  "z": normal = (0, 0, 1)
        elif strn == "-x": normal = (-1, 0, 0)
        elif strn == "-y": normal = (0, -1, 0)
        elif strn == "-z": normal = (0, 0, -1)
        plane = vtki.new("Plane")
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        cc = vtki.new("Cutter")
        cc.SetInputData(self.dataset)
        cc.SetCutFunction(plane)
        cc.Update()
        msh = Mesh(cc.GetOutput()).flat().lighting("ambient")
        msh.copy_properties_from(self)
        msh.pipeline = utils.OperationNode("slice", c="#edabab", parents=[self])
        return msh


##########################################################################
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

    def clone(self, deep=True) -> "RectilinearGrid":
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

    def isosurface(self, value=None) -> "vedo.Mesh":
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

    def cut_with_plane(self, origin=(0, 0, 0), normal="x") -> "UnstructuredGrid":
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

    def cut_with_mesh(self, mesh, invert=False, whole_cells=False, on_boundary=False) -> "UnstructuredGrid":
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


##########################################################################
class StructuredGrid(PointAlgorithms, MeshVisual):
    """
    Build a structured grid.
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
            inputobj : (vtkStructuredGrid, list, str)
                list of points and indices, or filename

        Example:
            ```python
            from vedo import *
            sgrid = StructuredGrid(dataurl+"structgrid.vts")
            print(sgrid)
            msh = sgrid.tomesh().lw(1)
            show(msh, axes=1, viewup="z")
            ```

            ```python
            from vedo import *

            cx = np.sqrt(np.linspace(100, 400, 10))
            cy = np.linspace(30, 40, 20)
            cz = np.linspace(40, 50, 30)
            x, y, z = np.meshgrid(cx, cy, cz)

            sgrid1 = StructuredGrid([x, y, z])
            sgrid1.cmap("viridis", sgrid1.coordinates[:, 0])
            print(sgrid1)

            sgrid2 = sgrid1.clone().cut_with_plane(normal=(-1,1,1), origin=[14,34,44])
            msh2 = sgrid2.tomesh(shrink=0.9).lw(1).cmap("viridis")

            show(
                [["StructuredGrid", sgrid1], ["Shrinked Mesh", msh2]],
                N=2, axes=1, viewup="z",
            )
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

        self.name = "StructuredGrid"
        self.filename = ""

        self.info = {}
        self.time =  time.time()

        ###############################
        if inputobj is None:
            self.dataset = vtki.vtkStructuredGrid()

        elif isinstance(inputobj, vtki.vtkStructuredGrid):
            self.dataset = inputobj

        elif isinstance(inputobj, StructuredGrid):
            self.dataset = inputobj.dataset

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            if inputobj.endswith(".vts"):
                reader = vtki.new("XMLStructuredGridReader")
            else:
                reader = vtki.new("StructuredGridReader")
            self.filename = inputobj
            reader.SetFileName(inputobj)
            reader.Update()
            self.dataset = reader.GetOutput()

        elif utils.is_sequence(inputobj):
            self.dataset = vtki.vtkStructuredGrid()
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
            vedo.logger.error(f"StructuredGrid: cannot understand input type {type(inputobj)}")
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
        """Print a summary for the `StructuredGrid` object."""
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
        HTML representation of the StructuredGrid object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.grids.StructuredGrid"
        help_url = "https://vedo.embl.es/docs/vedo/grids.html#StructuredGrid"

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

    def clone(self, deep=True) -> "StructuredGrid":
        """Return a clone copy of the StructuredGrid. Alias of `copy()`."""
        if deep:
            newrg = vtki.vtkStructuredGrid()
            newrg.CopyStructure(self.dataset)
            newrg.CopyAttributes(self.dataset)
            newvol = StructuredGrid(newrg)
        else:
            newvol = StructuredGrid(self.dataset)

        prop = vtki.vtkProperty()
        prop.DeepCopy(self.properties)
        newvol.actor.SetProperty(prop)
        newvol.properties = prop
        newvol.pipeline = utils.OperationNode("clone", parents=[self], c="#bbd0ff", shape="diamond")
        return newvol

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

    def cut_with_plane(self, origin=(0, 0, 0), normal="x") -> "vedo.UnstructuredGrid":
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

    def cut_with_mesh(self, mesh: Mesh, invert=False, whole_cells=False, on_boundary=False) -> "UnstructuredGrid":
        """
        Cut a `RectilinearGrid` with a `Mesh`.

        Use `invert` to return cut off part of the input object.

        Returns an `UnstructuredGrid` object.
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

    def isosurface(self, value=None) -> "vedo.Mesh":
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
 


######### WIP ##############
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
    
    def clone(self, deep=True) -> "ExplicitStructuredGrid":
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
    
    def cut_with_plane(self, origin=(0, 0, 0), normal="x") -> "vedo.UnstructuredGrid":
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
    
