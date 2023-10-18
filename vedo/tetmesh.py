#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import numpy as np
import vedo
from vedo import utils
from vedo.core import UGridAlgorithms
from vedo.mesh import Mesh
from vedo.file_io import download, loadUnStructuredGrid
from vedo.visual import VolumeVisual
from vedo.transformations import LinearTransform

__docformat__ = "google"

__doc__ = """
Work with tetrahedral meshes.

![](https://vedo.embl.es/images/volumetric/82767107-2631d500-9e25-11ea-967c-42558f98f721.jpg)
"""

__all__ = ["TetMesh", "delaunay3d"]


##########################################################################
def delaunay3d(mesh, radius=0, tol=None):
    """
    Create 3D Delaunay triangulation of input points.

    Arguments:

        radius : (float)
            specify distance (or "alpha") value to control output.
            For a non-zero values, only tetra contained within the circumsphere
            will be output.

        tol : (float)
            Specify a tolerance to control discarding of closely spaced points.
            This tolerance is specified as a fraction of the diagonal length of
            the bounding box of the points.
    """
    deln = vtk.vtkDelaunay3D()
    if utils.is_sequence(mesh):
        pd = vtk.vtkPolyData()
        vpts = vtk.vtkPoints()
        vpts.SetData(utils.numpy2vtk(mesh, dtype=np.float32))
        pd.SetPoints(vpts)
        deln.SetInputData(pd)
    else:
        deln.SetInputData(mesh.dataset)
    deln.SetAlpha(radius)
    deln.AlphaTetsOn()
    deln.AlphaTrisOff()
    deln.AlphaLinesOff()
    deln.AlphaVertsOff()
    deln.BoundingTriangulationOff()
    if tol:
        deln.SetTolerance(tol)
    deln.Update()
    m = TetMesh(deln.GetOutput())
    m.pipeline = utils.OperationNode(
        "delaunay3d", c="#e9c46a:#edabab", parents=[mesh],
    )
    return m


def _buildtetugrid(points, cells):
    ug = vtk.vtkUnstructuredGrid()

    if len(points) == 0:
        return ug
    if not utils.is_sequence(points[0]):
        return ug

    if len(cells) == 0:
        return ug

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
    ug.SetPoints(source_points)

    source_tets = vtk.vtkCellArray()
    for f in cells:
        ele = vtk.vtkTetra()
        pid = ele.GetPointIds()
        for i, fi in enumerate(f):
            pid.SetId(i, fi)
        source_tets.InsertNextCell(ele)
    ug.SetCells(vtk.VTK_TETRA, source_tets)
    return ug


##########################################################################
class TetMesh(VolumeVisual, UGridAlgorithms):
    """The class describing tetrahedral meshes."""

    def __init__(
        self,
        inputobj=None,
        c=("r", "y", "lg", "lb", "b"),  # ('b','lb','lg','y','r')
        alpha=(0.5, 1),
        alpha_unit=1,
        mapper="tetra",
    ):
        """
        Arguments:
            inputobj : (vtkDataSet, list, str)
                list of points and tet indices, or filename
            alpha_unit : (float)
                opacity scale
            mapper : (str)
                choose a visualization style in `['tetra', 'raycast', 'zsweep']`
        """
        super().__init__()

        self.actor = vtk.vtkVolume()

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

        elif isinstance(inputobj, vedo.UGrid):
            self.dataset = inputobj.dataset

        elif isinstance(inputobj, vtk.vtkRectilinearGrid):
            r2t = vtk.vtkRectilinearGridToTetrahedra()
            r2t.SetInputData(inputobj)
            r2t.RememberVoxelIdOn()
            r2t.SetTetraPerCellTo6()
            r2t.Update()
            self.dataset = r2t.GetOutput()

        elif isinstance(inputobj, vtk.vtkDataSet):
            r2t = vtk.vtkDataSetTriangleFilter()
            r2t.SetInputData(inputobj)
            # r2t.TetrahedraOnlyOn()
            r2t.Update()
            self.dataset = r2t.GetOutput()

        elif isinstance(inputobj, str):
            self.filename = inputobj
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            ug = loadUnStructuredGrid(inputobj)
            tt = vtk.vtkDataSetTriangleFilter()
            tt.SetInputData(ug)
            tt.SetTetrahedraOnly(True)
            tt.Update()
            self.dataset = tt.GetOutput()

        elif utils.is_sequence(inputobj):
            self.dataset = _buildtetugrid(inputobj[0], inputobj[1])

        ###################
        if "tetra" in mapper:
            self.mapper = vtk.vtkProjectedTetrahedraMapper()
        elif "ray" in mapper:
            self.mapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()
        elif "zs" in mapper:
            self.mapper = vtk.vtkUnstructuredGridVolumeZSweepMapper()
        elif isinstance(mapper, vtk.vtkMapper):
            self.mapper = mapper
        else:
            vedo.logger.error(f"Unknown mapper type {type(mapper)}")
            raise RuntimeError()

        self.properties = self.actor.GetProperty()

        self.mapper.SetInputData(self.dataset)
        self.actor.SetMapper(self.mapper)
        self.cmap(c).alpha(alpha)
        if alpha_unit:
            self.properties.SetScalarOpacityUnitDistance(alpha_unit)

        # remember stuff:
        self._color = c
        self._alpha = alpha
        self._alpha_unit = alpha_unit

        self.pipeline = utils.OperationNode(
            self, comment=f"#tets {self.dataset.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        # -----------------------------------------------------------

    def __str__(self):
        """Print a string summary of the `TetMesh` object."""
        opts = dict(c='m', return_string=True)
        bnds = self.bounds()
        ug = self.dataset
        bx1, bx2 = utils.precision(bnds[0], 3), utils.precision(bnds[1], 3)
        by1, by2 = utils.precision(bnds[2], 3), utils.precision(bnds[3], 3)
        bz1, bz2 = utils.precision(bnds[4], 3), utils.precision(bnds[5], 3)
        s = vedo.printc("TetMesh".ljust(70), bold=True, invert=True, **opts)
        s+= vedo.printc("nr. of tetras".ljust(14) + ": ", bold=True, end="", **opts)
        s+= vedo.printc(ug.GetNumberOfCells(), bold=False, **opts)
        s+= vedo.printc("bounds".ljust(14) + ": ", bold=True, end="", **opts)
        s+= vedo.printc("x=(" + bx1 + ", " + bx2 + ")", bold=False, end="", **opts)
        s+= vedo.printc(" y=(" + by1 + ", " + by2 + ")", bold=False, end="", **opts)
        s+= vedo.printc(" z=(" + bz1 + ", " + bz2 + ")", bold=False, **opts)
        # _print_data(ug, cf) #TODO
        return s

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


    def clone(self, mapper="tetra"):
        """Clone the `TetMesh` object to yield an exact copy."""
        ug = vtk.vtkUnstructuredGrid()
        ug.DeepCopy(self.dataset)

        cloned = TetMesh(ug, mapper=mapper)
        pr = vtk.vtkVolumeProperty()
        pr.DeepCopy(self.properties)
        cloned.actor.SetProperty(pr)

        cloned.mapper.SetScalarMode(self.mapper.GetScalarMode())

        cloned.pipeline = utils.OperationNode(
            "clone", c="#edabab", shape="diamond", parents=[self],
        )
        return cloned

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
        qf = vtk.vtkMeshQuality()
        qf.SetInputData(self.dataset)
        qf.SetTetQualityMeasure(metric)
        qf.SaveCellQualityOn()
        qf.Update()
        self._update(qf.GetOutput())
        return utils.vtk2numpy(qf.GetOutput().GetCellData().GetArray("Quality"))

    def compute_tets_volume(self):
        """Add to this mesh a cell data array containing the tetrahedron volume."""
        csf = vtk.vtkCellSizeFilter()
        csf.SetInputData(self.dataset)
        csf.SetComputeArea(False)
        csf.SetComputeVolume(True)
        csf.SetComputeLength(False)
        csf.SetComputeVertexCount(False)
        csf.SetVolumeArrayName("TetVolume")
        csf.Update()
        self._update(csf.GetOutput())
        return utils.vtk2numpy(csf.GetOutput().GetCellData().GetArray("TetVolume"))

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
        vald = vtk.vtkCellValidator()
        if tol:
            vald.SetTolerance(tol)
        vald.SetInputData(self.dataset)
        vald.Update()
        varr = vald.GetOutput().GetCellData().GetArray("ValidityState")
        return utils.vtk2numpy(varr)

    def threshold(self, name=None, above=None, below=None, on="cells"):
        """
        Threshold the tetrahedral mesh by a cell scalar value.
        Reduce to only tets which satisfy the threshold limits.

        - if `above = below` will only select tets with that specific value.
        - if `above > below` selection range is flipped.

        Set keyword "on" to either "cells" or "points".
        """
        th = vtk.vtkThreshold()
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
        decimate = vtk.vtkUnstructuredGridQuadricDecimation()
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
        sd = vtk.vtkSubdivideTetra()
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
        cf = vtk.vtkContourFilter()  # vtk.vtkContourGrid()
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

        clp = vtk.vtkCleanPolyData()
        clp.SetInputData(cf.GetOutput())
        clp.Update()
        msh = Mesh(clp.GetOutput(), c=None).phong()
        msh.mapper.SetLookupTable(utils.ctf2lut(self))
        msh.pipeline = utils.OperationNode("isosurface", c="#edabab", parents=[self])
        return msh

    def slice(self, origin=(0, 0, 0), normal=(1, 0, 0)):
        """
        Return a 2D slice of the mesh by a plane passing through origin and assigned normal.
        """
        strn = str(normal)
        if strn   ==  "x": normal = (1, 0, 0)
        elif strn ==  "y": normal = (0, 1, 0)
        elif strn ==  "z": normal = (0, 0, 1)
        elif strn == "-x": normal = (-1, 0, 0)
        elif strn == "-y": normal = (0, -1, 0)
        elif strn == "-z": normal = (0, 0, -1)
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        cc = vtk.vtkCutter()
        cc.SetInputData(self.dataset)
        cc.SetCutFunction(plane)
        cc.Update()
        msh = Mesh(cc.GetOutput()).flat().lighting("ambient")
        msh.mapper.SetLookupTable(utils.ctf2lut(self))
        msh.pipeline = utils.OperationNode("slice", c="#edabab", parents=[self])
        return msh
