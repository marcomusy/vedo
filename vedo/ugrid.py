#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import vedo.vtkclasses as vtk

import vedo
from vedo import utils
from vedo.core import UGridAlgorithms
from vedo.file_io import download, loadUnStructuredGrid
from vedo.visual import VolumeVisual


__docformat__ = "google"

__doc__ = """
Work with unstructured grid datasets
"""

__all__ = ["UGrid"]

#########################################################################
class UGrid(VolumeVisual, UGridAlgorithms):
    """Support for UnstructuredGrid objects."""

    def __init__(self, inputobj=None):
        """
        Support for UnstructuredGrid objects.

        Arguments:
            inputobj : (list, vtkUnstructuredGrid, str)
                A list in the form `[points, cells, celltypes]`,
                or a vtkUnstructuredGrid object, or a filename

        Celltypes are identified by the following convention:
            - VTK_TETRA = 10
            - VTK_VOXEL = 11
            - VTK_HEXAHEDRON = 12
            - VTK_WEDGE = 13
            - VTK_PYRAMID = 14
            - VTK_HEXAGONAL_PRISM = 15
            - VTK_PENTAGONAL_PRISM = 16
        """
        super().__init__()

        self.dataset = None
        self.actor = vtk.vtkVolume()
        self.properties = self.actor.GetProperty()

        self.name = "UGrid"
        self.filename = ""

        ###################
        inputtype = str(type(inputobj))

        if inputobj is None:
            self.dataset = vtk.vtkUnstructuredGrid()

        elif utils.is_sequence(inputobj):

            pts, cells, celltypes = inputobj
            assert len(cells) == len(celltypes)

            self.dataset = vtk.vtkUnstructuredGrid()

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
            points = vtk.vtkPoints()
            points.SetData(vpts)
            self.dataset.SetPoints(points)

            # This fill the points and use cells to define orientation
            # points = vtk.vtkPoints()
            # for c in cells:
            #       for pid in c:
            #           points.InsertNextPoint(pts[pid])
            # self.dataset.SetPoints(points)

            # Fill cells
            # https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
            for i, ct in enumerate(celltypes):
                cell_conn = cells[i]
                if ct == vtk.VTK_HEXAHEDRON:
                    cell = vtk.vtkHexahedron()
                elif ct == vtk.VTK_TETRA:
                    cell = vtk.vtkTetra()
                elif ct == vtk.VTK_VOXEL:
                    cell = vtk.vtkVoxel()
                elif ct == vtk.VTK_WEDGE:
                    cell = vtk.vtkWedge()
                elif ct == vtk.VTK_PYRAMID:
                    cell = vtk.vtkPyramid()
                elif ct == vtk.VTK_HEXAGONAL_PRISM:
                    cell = vtk.vtkHexagonalPrism()
                elif ct == vtk.VTK_PENTAGONAL_PRISM:
                    cell = vtk.vtkPentagonalPrism()
                else:
                    print("UGrid: cell type", ct, "not supported. Skip.")
                    continue
                cpids = cell.GetPointIds()
                for j, pid in enumerate(cell_conn):
                    cpids.SetId(j, pid)
                self.dataset.InsertNextCell(ct, cpids)

        elif "UnstructuredGrid" in inputtype:
            self.dataset = inputobj

        elif isinstance(inputobj, str):
            self.filename = inputobj
            if "https://" in inputobj:
                self.filename = inputobj
                inputobj = download(inputobj, verbose=False)
            self.dataset = loadUnStructuredGrid(inputobj)

        else:
            vedo.logger.error(f"cannot understand input type {inputtype}")
            return

        self.mapper = vtk.new("UnstructuredGridVolumeRayCastMapper")
        self.actor.SetMapper(self.mapper)

        self.mapper.SetInputData(self.dataset) ###NOT HERE?

        self.pipeline = utils.OperationNode(
            self, comment=f"#cells {self.dataset.GetNumberOfCells()}",
            c="#4cc9f0",
        )

    # ------------------------------------------------------------------

    def __str__(self):
        """Print a string summary of the `UGrid` object."""
        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(self.memory_address())})".ljust(75),
            c="m", bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m\u001b[35m"

        out += "nr. of verts".ljust(14) + ": " + str(self.npoints) + "\n"
        out += "nr. of cells".ljust(14)+ ": " + str(self.ncells) + "\n"

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
            if self.dataset.GetPointData().GetScalars().GetName() == key:
                mark_active += " *"
            elif self.dataset.GetPointData().GetVectors().GetName() == key:
                mark_active += " **"
            elif self.dataset.GetPointData().GetTensors().GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), ndim={arr.ndim}'
            out += f", range=({rng})\n"

        for key in self.celldata.keys():
            arr = self.celldata[key]
            rng = utils.precision(arr.min(), 3) + ", " + utils.precision(arr.max(), 3)
            mark_active = "celldata"
            if self.dataset.GetCellData().GetScalars().GetName() == key:
                mark_active += " *"
            elif self.dataset.GetCellData().GetVectors().GetName() == key:
                mark_active += " **"
            elif self.dataset.GetCellData().GetTensors().GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), ndim={arr.ndim}'
            out += f", range=({rng})\n"

        for key in self.metadata.keys():
            arr = self.metadata[key]
            out+= "metadata".ljust(14) + ": " + f'"{key}" ({len(arr)} values)\n'

        return out.rstrip() + "\x1b[0m"

    def print(self):
        """Print a description of the UGrid."""
        print(self.__str__())
        return self


    def _repr_html_(self):
        """
        HTML representation of the UGrid object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.ugrid.UGrid"
        help_url = "https://vedo.embl.es/docs/vedo/ugrid.html"

        # self.mapper.SetInputData(self.dataset)
        arr = self.thumbnail()
        im = Image.fromarray(arr)
        buffered = io.BytesIO()
        im.save(buffered, format="PNG", quality=100)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = "data:image/png;base64," + encoded
        image = f"<img src='{url}'></img>"

        bounds = "<br/>".join(
            [
                utils.precision(min_x,4) + " ... " + utils.precision(max_x,4)
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

        pts = self.vertices
        cm = np.mean(pts, axis=0)

        all = [
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
        return "\n".join(all)

    def copy(self, deep=True):
        """Return a copy of the object. Alias of `clone()`."""
        return self.clone(deep=deep)

    def clone(self, deep=True):
        """Clone the UGrid object to yield an exact copy."""
        ug = vtk.vtkUnstructuredGrid()
        if deep:
            ug.DeepCopy(self.dataset)
        else:
            ug.ShallowCopy(self.dataset)

        cloned = UGrid(ug)
        pr = vtk.vtkVolumeProperty()
        pr.DeepCopy(self.properties)
        cloned.actor.SetProperty(pr)
        cloned.properties = pr

        cloned.pipeline = utils.OperationNode(
            "clone", parents=[self], shape='diamond', c='#bbe1ed',
        )
        return cloned

    def extract_cell_type(self, ctype):
        """Extract a specific cell type and return a new `UGrid`."""
        uarr = self.dataset.GetCellTypesArray()
        ctarrtyp = np.where(utils.vtk2numpy(uarr) == ctype)[0]
        uarrtyp = utils.numpy2vtk(ctarrtyp, deep=False, dtype="id")
        selection_node = vtk.new("SelectionNode")
        selection_node.SetFieldType(vtk.get_class("SelectionNode").CELL)
        selection_node.SetContentType(vtk.get_class("SelectionNode").INDICES)
        selection_node.SetSelectionList(uarrtyp)
        selection = vtk.new("Selection")
        selection.AddNode(selection_node)
        es = vtk.new("ExtractSelection")
        es.SetInputData(0, self.dataset)
        es.SetInputData(1, selection)
        es.Update()

        ug = UGrid(es.GetOutput())

        ug.pipeline = utils.OperationNode(
            "extract_cell_type", comment=f"type {ctype}",
            c="#edabab", parents=[self],
        )
        return ug