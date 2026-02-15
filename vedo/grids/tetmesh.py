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
from vedo.transformations import LinearTransform
from .unstructured import UnstructuredGrid

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

        elif isinstance(inputobj, vtki.vtkDataSet) or (
            hasattr(inputobj, "dataset") and inputobj.dataset
        ):
            r2t = vtki.new("DataSetTriangleFilter")
            try:
                r2t.SetInputData(inputobj)
            except TypeError:
                r2t.SetInputData(inputobj.dataset)

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

            for f in cells:
                ele = vtki.vtkTetra()
                pid = ele.GetPointIds()
                for i, fi in enumerate(f):
                    pid.SetId(i, fi)
                self.dataset.InsertNextCell(vtki.cell_types["TETRA"], pid)

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

    def generate_random_points(self, n, min_radius=0) -> vedo.Points:
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

    def isosurface(self, value=True, flying_edges=None) -> vedo.Mesh:
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

    def slice(self, origin=(0, 0, 0), normal=(1, 0, 0)) -> vedo.Mesh:
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

