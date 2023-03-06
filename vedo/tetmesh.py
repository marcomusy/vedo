#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import vedo
from vedo import utils
from vedo.base import BaseGrid
from vedo.mesh import Mesh
from vedo.io import download, loadUnStructuredGrid


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
        deln.SetInputData(mesh.GetMapper().GetInput())
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

    sourceTets = vtk.vtkCellArray()
    for f in cells:
        ele = vtk.vtkTetra()
        pid = ele.GetPointIds()
        for i, fi in enumerate(f):
            pid.SetId(i, fi)
        sourceTets.InsertNextCell(ele)
    ug.SetCells(vtk.VTK_TETRA, sourceTets)
    return ug


##########################################################################
class TetMesh(BaseGrid, vtk.vtkVolume):
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
        vtk.vtkVolume.__init__(self)
        BaseGrid.__init__(self)

        self.useArray = 0

        # inputtype = str(type(inputobj))
        # print('TetMesh inputtype', inputtype)

        ###################
        if inputobj is None:
            self._data = vtk.vtkUnstructuredGrid()

        elif isinstance(inputobj, vtk.vtkUnstructuredGrid):
            self._data = inputobj

        elif isinstance(inputobj, vtk.vtkRectilinearGrid):
            r2t = vtk.vtkRectilinearGridToTetrahedra()
            r2t.SetInputData(inputobj)
            r2t.RememberVoxelIdOn()
            r2t.SetTetraPerCellTo6()
            r2t.Update()
            self._data = r2t.GetOutput()

        elif isinstance(inputobj, vtk.vtkDataSet):
            r2t = vtk.vtkDataSetTriangleFilter()
            r2t.SetInputData(inputobj)
            # r2t.TetrahedraOnlyOn()
            r2t.Update()
            self._data = r2t.GetOutput()

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            ug = loadUnStructuredGrid(inputobj)
            tt = vtk.vtkDataSetTriangleFilter()
            tt.SetInputData(ug)
            tt.SetTetrahedraOnly(True)
            tt.Update()
            self._data = tt.GetOutput()

        elif utils.is_sequence(inputobj):
            # if "ndarray" not in inputtype:
            #     inputobj = np.array(inputobj)
            self._data = _buildtetugrid(inputobj[0], inputobj[1])

        ###################
        if "tetra" in mapper:
            self._mapper = vtk.vtkProjectedTetrahedraMapper()
        elif "ray" in mapper:
            self._mapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()
        elif "zs" in mapper:
            self._mapper = vtk.vtkUnstructuredGridVolumeZSweepMapper()
        elif isinstance(mapper, vtk.vtkMapper):
            self._mapper = mapper
        else:
            vedo.logger.error(f"Unknown mapper type {type(mapper)}")
            raise RuntimeError()

        self._mapper.SetInputData(self._data)
        self.SetMapper(self._mapper)
        self.color(c).alpha(alpha)
        if alpha_unit:
            self.GetProperty().SetScalarOpacityUnitDistance(alpha_unit)

        # remember stuff:
        self._color = c
        self._alpha = alpha
        self._alpha_unit = alpha_unit

        self.pipeline = utils.OperationNode(
            self, comment=f"#tets {self._data.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        # -----------------------------------------------------------

    def _update(self, data):
        self._data = data
        self.mapper().SetInputData(data)
        self.mapper().Modified()
        return self

    def clone(self):
        """Clone the `TetMesh` object to yield an exact copy."""
        ugCopy = vtk.vtkUnstructuredGrid()
        ugCopy.DeepCopy(self._data)

        cloned = TetMesh(ugCopy)
        pr = vtk.vtkVolumeProperty()
        pr.DeepCopy(self.GetProperty())
        cloned.SetProperty(pr)

        # assign the same transformation to the copy
        cloned.SetOrigin(self.GetOrigin())
        cloned.SetScale(self.GetScale())
        cloned.SetOrientation(self.GetOrientation())
        cloned.SetPosition(self.GetPosition())

        cloned.mapper().SetScalarMode(self.mapper().GetScalarMode())
        cloned.name = self.name

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
        qf.SetInputData(self._data)
        qf.SetTetQualityMeasure(metric)
        qf.SaveCellQualityOn()
        qf.Update()
        self._update(qf.GetOutput())
        return utils.vtk2numpy(qf.GetOutput().GetCellData().GetArray("Quality"))

    def compute_tets_volume(self):
        """Add to this mesh a cell data array containing the tetrahedron volume."""
        csf = vtk.vtkCellSizeFilter()
        csf.SetInputData(self._data)
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
        vald.SetInputData(self._data)
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
        th.SetInputData(self._data)

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

        if above is not None and below is not None:
            if above > below:
                if vedo.vtk_version[0] >= 9:
                    th.SetInvert(True)
                    th.ThresholdBetween(below, above)
                else:
                    vedo.logger.error("in vtk<9, above cannot be larger than below. Skip")
                    return self
            else:
                th.ThresholdBetween(above, below)

        elif above is not None:
            th.ThresholdByUpper(above)

        elif below is not None:
            th.ThresholdByLower(below)

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
        decimate.SetInputData(self._data)
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
        sd.SetInputData(self._data)
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
        if not self._data.GetPointData().GetScalars():
            self.map_cells_to_points()
        scrange = self._data.GetPointData().GetScalars().GetRange()
        cf = vtk.vtkContourFilter()  # vtk.vtkContourGrid()
        cf.SetInputData(self._data)

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
        msh.mapper().SetLookupTable(utils.ctf2lut(self))
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
        cc.SetInputData(self._data)
        cc.SetCutFunction(plane)
        cc.Update()
        msh = Mesh(cc.GetOutput()).flat().lighting("ambient")
        msh.mapper().SetLookupTable(utils.ctf2lut(self))
        msh.pipeline = utils.OperationNode("slice", c="#edabab", parents=[self])
        return msh
