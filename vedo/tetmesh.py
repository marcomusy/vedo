#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from weakref import ref as weak_ref_to

import vedo.vtkclasses as vtk

import numpy as np
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

__all__ = ["UnstructuredGrid", "TetMesh"]

#########################################################################
class UnstructuredGrid(MeshVisual, PointAlgorithms):
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

        self.mapper = vtk.new("PolyDataMapper")
        self._actor = vtk.vtkActor()
        self._actor.retrieve_object = weak_ref_to(self)
        self._actor.SetMapper(self.mapper)
        self.properties = self._actor.GetProperty()

        self.transform = LinearTransform()

        self.name = "UnstructuredGrid"
        self.filename = ""
        self.info = {}
        self.time = 0
        self.rendered_at = set()

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
                    print("UnstructuredGrid: cell type", ct, "not supported. Skip.")
                    continue
                cpids = cell.GetPointIds()
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
                reader = vtk.new("XMLUnstructuredGridReader")
            else:
                reader = vtk.new("UnstructuredGridReader")
            self.filename = inputobj
            reader.SetFileName(inputobj)
            reader.Update()
            self.dataset = reader.GetOutput()

        else:
            vedo.logger.error(f"cannot understand input type {inputtype}")
            return

        self.pipeline = utils.OperationNode(
            self, comment=f"#cells {self.dataset.GetNumberOfCells()}",
            c="#4cc9f0",
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
            if   a_scalars and a_scalars.GetName() == key:
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
            if   a_scalars and a_scalars.GetName() == key:
                mark_active += " *"
            elif a_vectors and a_vectors.GetName() == key:
                mark_active += " **"
            elif a_tensors and a_tensors.GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), ndim={arr.ndim}'
            out += f", range=({rng})\n"

        for key in self.metadata.keys():
            arr = self.metadata[key]
            out+= "metadata".ljust(14) + ": " + f'"{key}" ({len(arr)} values)\n'

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

        library_name = "vedo.tetmesh.UnstructuredGrid"
        help_url = "https://vedo.embl.es/docs/vedo/tetmesh.html"

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

    @property
    def actor(self):
        """Return the `vtkActor` of the object."""
        # print("building actor")
        gf = vtk.new("GeometryFilter")
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
        # self.mapper.SetInputData(data)
        # self.mapper.Modified()
        ## self.actor.Modified()
        return self

    def copy(self, deep=True):
        """Return a copy of the object. Alias of `clone()`."""
        return self.clone(deep=deep)

    def clone(self, deep=True):
        """Clone the UnstructuredGrid object to yield an exact copy."""
        ug = vtk.vtkUnstructuredGrid()
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
            "clone", parents=[self], shape='diamond', c='#bbe1ed',
        )
        return cloned

    def bounds(self):
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        # OVERRIDE CommonAlgorithms.bounds() which is too slow
        return self.dataset.GetBounds()

    def threshold(self, name=None, above=None, below=None, on="cells"):
        """
        Threshold the tetrahedral mesh by a cell scalar value.
        Reduce to only tets which satisfy the threshold limits.

        - if `above = below` will only select tets with that specific value.
        - if `above > below` selection range is flipped.

        Set keyword "on" to either "cells" or "points".
        """
        th = vtk.new("Threshold")
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

    def isosurface(self, value=None, flying_edges=True):
        """
        Return an `Mesh` isosurface extracted from the `Volume` object.

        Set `value` as single float or list of values to draw the isosurface(s).
        Use flying_edges for faster results (but sometimes can interfere with `smooth()`).

        Examples:
            - [isosurfaces.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces.py)

                ![](https://vedo.embl.es/images/volumetric/isosurfaces.png)
        """
        scrange = self.dataset.GetScalarRange()

        if flying_edges:
            cf = vtk.new("FlyingEdges3D")
            cf.InterpolateAttributesOn()
        else:
            cf = vtk.new("ContourFilter")
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

    def tomesh(self, fill=True, shrink=1.0):
        """
        Build a polygonal `Mesh` from the current object.

        If `fill=True`, the interior faces of all the cells are created.
        (setting a `shrink` value slightly smaller than the default 1.0
        can avoid flickering due to internal adjacent faces).

        If `fill=False`, only the boundary faces will be generated.
        """
        gf = vtk.new("GeometryFilter")
        if fill:
            sf = vtk.new("ShrinkFilter")
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

    def extract_cell_by_type(self, ctype):
        """Extract a specific cell type and return a new `UnstructuredGrid`."""
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

        ug = UnstructuredGrid(es.GetOutput())

        ug.pipeline = utils.OperationNode(
            "extract_cell_type", comment=f"type {ctype}",
            c="#edabab", parents=[self],
        )
        return ug

    def extract_cells_by_id(self, idlist, use_point_ids=False):
        """Return a new `UnstructuredGrid` composed of the specified subset of indices."""
        selection_node = vtk.new("SelectionNode")
        if use_point_ids:
            selection_node.SetFieldType(vtk.get_class("SelectionNode").POINT)
            contcells = vtk.get_class("SelectionNode").CONTAINING_CELLS()
            selection_node.GetProperties().Set(contcells, 1)
        else:
            selection_node.SetFieldType(vtk.get_class("SelectionNode").CELL)
        selection_node.SetContentType(vtk.get_class("SelectionNode").INDICES)
        vidlist = utils.numpy2vtk(idlist, dtype="id")
        selection_node.SetSelectionList(vidlist)
        selection = vtk.new("Selection")
        selection.AddNode(selection_node)
        es = vtk.new("ExtractSelection")
        es.SetInputData(0, self)
        es.SetInputData(1, selection)
        es.Update()

        ug = vedo.tetmesh.UnstructuredGrid(es.GetOutput())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.properties)
        ug.SetProperty(pr)
        ug.properties = pr

        ug.mapper.SetLookupTable(utils.ctf2lut(self))
        ug.pipeline = utils.OperationNode(
            "extract_cells_by_id",
            parents=[self],
            comment=f"#cells {self.dataset.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        return ug

    def find_cell(self, p):
        """Locate the cell that contains a point and return the cell ID."""
        cell = vtk.vtkTetra()
        cell_id = vtk.mutable(0)
        tol2 = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        pcoords = [0, 0, 0]
        weights = [0, 0, 0]
        cid = self.dataset.FindCell(
            p, cell, cell_id, tol2, sub_id, pcoords, weights)
        return cid

    def clean(self):
        """
        Cleanup unused points and empty cells
        """
        cl = vtk.new("StaticCleanUnstructuredGrid")
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

    def extract_cells_on_plane(self, origin, normal):
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtk.new("3DLinearGridCrinkleExtractor")
        bf.SetInputData(self.dataset)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        plane = vtk.new("Plane")
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

    def extract_cells_on_sphere(self, center, radius):
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtk.new("3DLinearGridCrinkleExtractor")
        bf.SetInputData(self.dataset)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        sph = vtk.new("Sphere")
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

    def extract_cells_on_cylinder(self, center, axis, radius):
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtk.new("3DLinearGridCrinkleExtractor")
        bf.SetInputData(self.dataset)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        cyl = vtk.new("Cylinder")
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
    
    def cut_with_plane(self, origin=(0, 0, 0), normal="x"):
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
        plane = vtk.new("Plane")
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        clipper = vtk.new("ClipDataSet")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(plane)
        clipper.GenerateClipScalarsOff()
        clipper.GenerateClippedOutputOff()
        clipper.SetValue(0)
        clipper.Update()
        cout = clipper.GetOutput()

        if isinstance(cout, vtk.vtkUnstructuredGrid):
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

    def cut_with_box(self, box):
        """
        Cut the grid with the specified bounding box.

        Parameter box has format [xmin, xmax, ymin, ymax, zmin, zmax].
        If an object is passed, its bounding box are used.

        This method always returns a TetMesh object.

        Example:
        ```python
        from vedo import *
        tetmesh = TetMesh(dataurl+'limb_ugrid.vtk')
        tetmesh.color('rainbow')
        cu = Cube(side=500).x(500) # any Mesh works
        tetmesh.cut_with_box(cu).show(axes=1)
        ```

        ![](https://vedo.embl.es/images/feats/tet_cut_box.png)
        """
        bc = vtk.new("BoxClipDataSet")
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


    def cut_with_mesh(
            self, mesh, invert=False, whole_cells=False, on_boundary=False
        ):
        """
        Cut a UnstructuredGrid or TetMesh with a Mesh.

        Use `invert` to return cut off part of the input object.
        """
        ug = self.dataset

        ippd = vtk.new("ImplicitPolyDataDistance")
        ippd.SetInput(mesh.dataset)

        if whole_cells or on_boundary:
            clipper = vtk.new("ExtractGeometry")
            clipper.SetInputData(ug)
            clipper.SetImplicitFunction(ippd)
            clipper.SetExtractInside(not invert)
            clipper.SetExtractBoundaryCells(False)
            if on_boundary:
                clipper.SetExtractBoundaryCells(True)
                clipper.SetExtractOnlyBoundaryCells(True)
        else:
            signed_dists = vtk.vtkFloatArray()
            signed_dists.SetNumberOfComponents(1)
            signed_dists.SetName("SignedDistance")
            for pointId in range(ug.GetNumberOfPoints()):
                p = ug.GetPoint(pointId)
                signed_dist = ippd.EvaluateFunction(p)
                signed_dists.InsertNextValue(signed_dist)
            ug.GetPointData().AddArray(signed_dists)
            ug.GetPointData().SetActiveScalars("SignedDistance") # NEEDED
            clipper = vtk.new("ClipDataSet")
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
            inputobj : (vtkDataSet, list, str)
                list of points and tet indices, or filename
        """
        super().__init__()

        self.dataset = None

        self.mapper = vtk.new("PolyDataMapper")
        self._actor = vtk.vtkActor()
        self._actor.retrieve_object = weak_ref_to(self)
        self._actor.SetMapper(self.mapper)
        self.properties = self._actor.GetProperty()

        self.transform = LinearTransform()

        self.name = "TetMesh"
        self.filename = ""

        # inputtype = str(type(inputobj))
        # print('TetMesh inputtype', inputtype)

        ###################
        if inputobj is None:
            self.dataset = vtk.vtkUnstructuredGrid()

        elif isinstance(inputobj, vtk.vtkUnstructuredGrid):
            self.dataset = inputobj

        elif isinstance(inputobj, UnstructuredGrid):
            self.dataset = inputobj.dataset

        elif isinstance(inputobj, vtk.vtkRectilinearGrid):
            r2t = vtk.new("RectilinearGridToTetrahedra")
            r2t.SetInputData(inputobj)
            r2t.RememberVoxelIdOn()
            r2t.SetTetraPerCellTo6()
            r2t.Update()
            self.dataset = r2t.GetOutput()

        elif isinstance(inputobj, vtk.vtkDataSet):
            r2t = vtk.new("DataSetTriangleFilter")
            r2t.SetInputData(inputobj)
            r2t.TetrahedraOnlyOn()
            r2t.Update()
            self.dataset = r2t.GetOutput()

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            if inputobj.endswith(".vtu"):
                reader = vtk.new("XMLUnstructuredGridReader")
            else:
                reader = vtk.new("UnstructuredGridReader")
            self.filename = inputobj
            reader.SetFileName(inputobj)
            reader.Update()
            ug = reader.GetOutput()
            tt = vtk.new("DataSetTriangleFilter")
            tt.SetInputData(ug)
            tt.SetTetrahedraOnly(True)
            tt.Update()
            self.dataset = tt.GetOutput()

        elif utils.is_sequence(inputobj):            
            self.dataset = vtk.vtkUnstructuredGrid()

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

            source_points = vtk.vtkPoints()
            varr = utils.numpy2vtk(points, dtype=np.float32)
            source_points.SetData(varr)
            self.dataset.SetPoints(source_points)

            source_tets = vtk.vtkCellArray()
            for f in cells:
                ele = vtk.vtkTetra()
                pid = ele.GetPointIds()
                for i, fi in enumerate(f):
                    pid.SetId(i, fi)
                source_tets.InsertNextCell(ele)
            self.dataset.SetCells(vtk.VTK_TETRA, source_tets)

        else:
            vedo.logger.error(f"cannot understand input type {inputtype}")
            return

        self.pipeline = utils.OperationNode(
            self, comment=f"#tets {self.dataset.GetNumberOfCells()}",
            c="#9e2a2b",
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
        out += "nr. of tetras".ljust(14)+ ": " + str(self.ncells) + "\n"

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
            if   a_scalars and a_scalars.GetName() == key:
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
            if   a_scalars and a_scalars.GetName() == key:
                mark_active += " *"
            elif a_vectors and a_vectors.GetName() == key:
                mark_active += " **"
            elif a_tensors and a_tensors.GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), ndim={arr.ndim}'
            out += f", range=({rng})\n"

        for key in self.metadata.keys():
            arr = self.metadata[key]
            out+= "metadata".ljust(14) + ": " + f'"{key}" ({len(arr)} values)\n'

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

        library_name = "vedo.tetmesh.TetMesh"
        help_url = "https://vedo.embl.es/docs/vedo/tetmesh.html"

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

    def compute_quality(self, metric=7):
        """
        Calculate functions of quality for the elements of a triangular mesh.
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
        qf = vtk.new("MeshQuality")
        qf.SetInputData(self.dataset)
        qf.SetTetQualityMeasure(metric)
        qf.SaveCellQualityOn()
        qf.Update()
        self._update(qf.GetOutput())
        return utils.vtk2numpy(qf.GetOutput().GetCellData().GetArray("Quality"))

    def check_validity(self, tol=0):
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
        vald = vtk.new("CellValidator")
        if tol:
            vald.SetTolerance(tol)
        vald.SetInputData(self.dataset)
        vald.Update()
        varr = vald.GetOutput().GetCellData().GetArray("ValidityState")
        return utils.vtk2numpy(varr)

    def decimate(self, scalars_name, fraction=0.5, n=0):
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
        decimate = vtk.new("UnstructuredGridQuadricDecimation")
        decimate.SetInputData(self.dataset)
        decimate.SetScalarsName(scalars_name)

        if n:  # n = desired number of points
            decimate.SetNumberOfTetsOutput(n)
        else:
            decimate.SetTargetReduction(1 - fraction)
        decimate.Update()
        self._update(decimate.GetOutput())
        self.pipeline = utils.OperationNode(
            "decimate", comment=f"array: {scalars_name}",
            c="#edabab", parents=[self],
        )
        return self

    def subdvide(self):
        """
        Increase the number of tets of a `TetMesh`.
        Subdivide one tetrahedron into twelve for every tetra.
        """
        sd = vtk.new("SubdivideTetra")
        sd.SetInputData(self.dataset)
        sd.Update()
        self._update(sd.GetOutput())
        self.pipeline = utils.OperationNode(
            "subdvide", c="#edabab", parents=[self],
        )
        return self

    def isosurface(self, value=True):
        """
        Return a `vedo.Mesh` isosurface.

        Set `value` to a single value or list of values to compute the isosurface(s).
        """
        if not self.dataset.GetPointData().GetScalars():
            self.map_cells_to_points()
        scrange = self.dataset.GetPointData().GetScalars().GetRange()
        cf = vtk.new("ContourFilter")  # vtk.new("ContourGrid")
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

        clp = vtk.new("CleanPolyData")
        clp.SetInputData(cf.GetOutput())
        clp.Update()
        msh = Mesh(clp.GetOutput(), c=None).phong()
        msh.copy_properties_from(self)
        msh.pipeline = utils.OperationNode("isosurface", c="#edabab", parents=[self])
        return msh

    def slice(self, origin=(0, 0, 0), normal=(1, 0, 0)):
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
        plane = vtk.new("Plane")
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        cc = vtk.new("Cutter")
        cc.SetInputData(self.dataset)
        cc.SetCutFunction(plane)
        cc.Update()
        msh = Mesh(cc.GetOutput()).flat().lighting("ambient")
        msh.copy_properties_from(self)
        msh.pipeline = utils.OperationNode("slice", c="#edabab", parents=[self])
        return msh
