import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray
import vtkplotter.docs as docs
import vtkplotter.utils as utils
from vtkplotter.base import ActorBase
from vtkplotter.mesh import Mesh
from vtkplotter.colors import printc
import vtkplotter.colors as colors
import numpy as np

__doc__ = (
    """
Support for tetrahedral meshes.
"""
    + docs._defs
)

__all__ = ["TetMesh", "delaunay3D", "tetralize"]


##########################################################################
def delaunay3D(mesh, alphaPar=0, tol=None, boundary=False):
    """Create 3D Delaunay triangulation of input points."""
    deln = vtk.vtkDelaunay3D()
    if utils.isSequence(mesh):
        pd = vtk.vtkPolyData()
        vpts = vtk.vtkPoints()
        vpts.SetData(numpy_to_vtk(np.ascontiguousarray(mesh), deep=True))
        pd.SetPoints(vpts)
        deln.SetInputData(pd)
    else:
        deln.SetInputData(mesh.GetMapper().GetInput())
    deln.SetAlpha(alphaPar)
    deln.AlphaTetsOn()
    deln.AlphaTrisOff()
    deln.AlphaLinesOff()
    deln.AlphaVertsOff()
    if tol:
        deln.SetTolerance(tol)
    deln.SetBoundingTriangulation(boundary)
    deln.Update()
    m = TetMesh(deln.GetOutput())
    return m


def tetralize(dataset, tetsOnly=False):
    """Tetralize any type of dataset.
    If true will cull all 1D and 2D cells from the output.

    Return a TetMesh.

    Example:

        .. code-block:: python

            from vtkplotter import *
            ug = loadUnStructuredGrid(datadir+'ugrid.vtk')
            tmesh = tetralize(ug)
            tmesh.write('ugrid.vtu').show(axes=1)
    """
    tt = vtk.vtkDataSetTriangleFilter()
    tt.SetInputData(dataset)
    tt.SetTetrahedraOnly(tetsOnly)
    tt.Update()
    m = TetMesh(tt.GetOutput())
    return m


##########################################################################
class TetMesh(vtk.vtkVolume, ActorBase):
    """The class describing tetrahedral meshes."""

    def __init__(self, inputobj=None,
                 c=('r','y','lg','lb','b'), #('b','lb','lg','y','r')
                 alpha=(0.5, 1),
                 alphaUnit=1,
                 mapper='tetra',
                 ):

        vtk.vtkVolume.__init__(self)
        ActorBase.__init__(self)

        self._ugrid = None

        self.useCells = True
        self.useArray = 0

        inputtype = str(type(inputobj))
        #printc('TetMesh inputtype', inputtype)

        ###################
        if inputobj is None:
            self._ugrid = vtk.vtkUnstructuredGrid()
        elif isinstance(inputobj, vtk.vtkUnstructuredGrid):
            self._ugrid = inputobj
        elif isinstance(inputobj, vtk.vtkRectilinearGrid):
            r2t = vtk.vtkRectilinearGridToTetrahedra()
            r2t.SetInputData(inputobj)
            r2t.RememberVoxelIdOn()
            r2t.SetTetraPerCellTo6()
            r2t.Update()
            self._ugrid = r2t.GetOutput()
        elif isinstance(inputobj, vtk.vtkDataSet):
            r2t = vtk.vtkDataSetTriangleFilter()
            r2t.SetInputData(inputobj)
            #r2t.TetrahedraOnlyOn()
            r2t.Update()
            self._ugrid = r2t.GetOutput()
        elif isinstance(inputobj, str):
            from vtkplotter.vtkio import loadUnStructuredGrid
            self._ugrid = loadUnStructuredGrid(inputobj)
        elif utils.isSequence(inputobj):
            if "ndarray" not in inputtype:
                inputobj = np.array(inputobj)
            self._ugrid = self._buildugrid(inputobj[0], inputobj[1])

        ###################
        if 'tetra' in mapper:
            self._mapper = vtk.vtkProjectedTetrahedraMapper()
        elif 'ray' in mapper:
            self._mapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()
        elif 'zs' in mapper:
            self._mapper = vtk.vtkUnstructuredGridVolumeZSweepMapper()
        elif isinstance(mapper, vtk.vtkMapper):
            self._mapper = mapper
        else:
            printc('Unknown mapper type', [mapper], c=1)
            return

        self._mapper.SetInputData(self._ugrid)
        self.SetMapper(self._mapper)
        self.color(c).alpha(alpha)
        if alphaUnit:
            self.GetProperty().SetScalarOpacityUnitDistance(alphaUnit)

        # remember stuff:
        self._color = c
        self._alpha = alpha
        self._alphaUnit = alphaUnit
        #-----------------------------------------------------------

    def _update(self, data):
        self._ugrid = data
        self._mapper.SetInputData(data)
        self._mapper.Modified()
        return self


    def _buildugrid(self, points, cells):
        if len(points) == 0:
            return None
        if not utils.isSequence(points[0]):
            return None

        ug = vtk.vtkUnstructuredGrid()
        sourcePoints = vtk.vtkPoints()
        varr = numpy_to_vtk(np.ascontiguousarray(points), deep=True)
        sourcePoints.SetData(varr)
        ug.SetPoints(sourcePoints)

        sourceTets = vtk.vtkCellArray()
        for f in cells:
            ele = vtk.vtkTetra()
            pid = ele.GetPointIds()
            for i, fi in enumerate(f):
                pid.SetId(i, fi)
            sourceTets.InsertNextCell(ele)
        ug.SetCells(vtk.VTK_TETRA, sourceTets)
        return ug


    def toMesh(self, fill=True, shrink=1.0):
        """
        Build a polygonal Mesh from the current TetMesh.

        If fill=True, the interior traingular faces of all the tets are created.
        In this case setting a `shrink` value slightly smaller than 1.0
        can avoid flickering due to adjacent faces.
        If fill=False, only the boundary triangles are generated.
        """
        gf = vtk.vtkGeometryFilter()
        if fill:
            ugtmp = vtk.vtkUnstructuredGrid()
            ugtmp.DeepCopy(self._ugrid)
            sf = vtk.vtkShrinkFilter()
            sf.SetInputData(self._ugrid)
            sf.SetShrinkFactor(shrink)
            sf.Update()
            gf.SetInputData(sf.GetOutput())
            gf.Update()
            del ugtmp
        else:
            gf.SetInputData(self._ugrid)
            gf.Update()
        poly = gf.GetOutput()

        msh = Mesh(poly).flat()
        msh.scalarbar = self.scalarbar
        lut = utils.ctf2lut(self)
        msh._mapper.SetLookupTable(lut)
        if self.useCells:
            msh._mapper.SetScalarModeToUseCellData()
        else:
            msh._mapper.SetScalarModeToUsePointData()
        # msh._mapper.SetScalarRange(msh._mapper.GetScalarRange())
        # print(msh._mapper.GetScalarRange(), lut.GetRange())
        # msh._mapper.SetScalarRange()
        # msh.selectCellArray('chem_0')
        return msh


    def points(self, pts=None, transformed=True, copy=False):
        """
        Set/Get the vertex coordinates of the mesh.
        Argument can be an index, a set of indices
        or a complete new set of points to update the mesh.

        :param bool transformed: if `False` ignore any previous transformation
            applied to the mesh.
        :param bool copy: if `False` return the reference to the points
            so that they can be modified in place, otherwise a copy is built.
        """
        if pts is None: ### getter

            vpts = self._ugrid.GetPoints()
            if vpts:
                if copy:
                    return np.array(vtk_to_numpy(vpts.GetData()))
                else:
                    return vtk_to_numpy(vpts.GetData())
            else:
                return np.array([])

        elif (utils.isSequence(pts) and not utils.isSequence(pts[0])) or isinstance(pts, (int, np.integer)):
            #passing a list of indices or a single index
            return vtk_to_numpy(self.polydata(transformed).GetPoints().GetData())[pts]

        else:           ### setter

            if len(pts) == 3 and len(pts[0]) != 3:
                # assume plist is in the format [all_x, all_y, all_z]
                pts = np.stack((pts[0], pts[1], pts[2]), axis=1)
            vpts = self._ugrid.GetPoints()
            vpts.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
            self._ugrid.GetPoints().Modified()
            # reset mesh to identity matrix position/rotation:
            self.PokeMatrix(vtk.vtkMatrix4x4())
            return self


    def cells(self):
        """
        Get the tetrahedral face connectivity ids as a numpy array.
        The output format is: [[id0 ... idn], [id0 ... idm],  etc].
        """
        arr1d = vtk_to_numpy(self._ugrid.GetCells().GetData())
        if arr1d is None:
            return []

        #Get cell connettivity ids as a 1D array. vtk format is:
        #[nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        if len(arr1d) == 0:
            arr1d = vtk_to_numpy(self._polydata.GetStrips().GetData())
            if arr1d is None:
                return []
        i = 0
        conn = []
        n = len(arr1d)
        if n:
            while True:
                cell = [arr1d[i+k] for k in range(1, arr1d[i]+1)]
                conn.append(cell)
                i += arr1d[i]+1
                if i >= n:
                    break
        return conn


    def clone(self):
        """Clone the ``TetMesh`` object to yield an exact copy."""
        ugCopy = vtk.vtkUnstructuredGrid()
        ugCopy.DeepCopy(self._ugrid)

        cloned = TetMesh(ugCopy)
        pr = vtk.vtkVolumeProperty()
        pr.DeepCopy(self.GetProperty())
        cloned.SetProperty(pr)

        #assign the same transformation to the copy
        cloned.SetOrigin(self.GetOrigin())
        cloned.SetScale(self.GetScale())
        cloned.SetOrientation(self.GetOrientation())
        cloned.SetPosition(self.GetPosition())

        cloned._mapper.SetScalarMode(self._mapper.GetScalarMode())
        cloned.name = self.name
        return cloned


    def ugrid(self):
        """Return the ``vtkUnstructuredGrid`` object."""
        return self._ugrid


    def color(self, col):
        """
        Assign a color or a set of colors along the range of the scalar value.
        A single constant color can also be assigned.
        Any matplotlib color map name is also accepted, e.g. ``volume.color('jet')``.

        E.g.: say that your tets scalar runs from -3 to 6,
        and you want -3 to show red and 1.5 violet and 6 green, then just set:

        ``volume.color(['red', 'violet', 'green'])``
        """
        smin, smax = self._ugrid.GetScalarRange()
        ctf = self.GetProperty().GetRGBTransferFunction()
        ctf.RemoveAllPoints()
        self._color = col

        if utils.isSequence(col):
            for i, ci in enumerate(col):
                r, g, b = colors.getColor(ci)
                x = smin + (smax - smin) * i / (len(col) - 1)
                ctf.AddRGBPoint(x, r, g, b)
                #colors.printc('\tcolor at', round(x, 1),
                #              '\tset to', colors.getColorName((r, g, b)), c='w', bold=0)
        elif isinstance(col, str):
            if col in colors.colors.keys() or col in colors.color_nicks.keys():
                r, g, b = colors.getColor(col)
                ctf.AddRGBPoint(smin, r,g,b) # constant color
                ctf.AddRGBPoint(smax, r,g,b)
            elif colors._mapscales:
                for x in np.linspace(smin, smax, num=64, endpoint=True):
                    r,g,b = colors.colorMap(x, name=col, vmin=smin, vmax=smax)
                    ctf.AddRGBPoint(x, r, g, b)
        elif isinstance(col, int):
            r, g, b = colors.getColor(col)
            ctf.AddRGBPoint(smin, r,g,b) # constant color
            ctf.AddRGBPoint(smax, r,g,b)
        else:
            colors.printc("volume.color(): unknown input type:", col, c=1)
        return self

    def alpha(self, alpha):
        """
        Assign a set of tranparencies along the range of the scalar value.
        A single constant value can also be assigned.

        E.g.: say alpha=(0.0, 0.3, 0.9, 1) and the scalar range goes from -10 to 150.
        Then all tets with a value close to -10 will be completely transparent, tets at 1/4
        of the range will get an alpha equal to 0.3 and voxels with value close to 150
        will be completely opaque.

        As a second option one can set explicit (x, alpha_x) pairs to define the transfer function.
        E.g.: say alpha=[(-5, 0), (35, 0.4) (123,0.9)] and the scalar range goes from -10 to 150.
        Then all tets below -5 will be completely transparent, tets with a scalar value of 35
        will get an opacity of 40% and above 123 alpha is set to 90%.
        """
        smin, smax = self._ugrid.GetScalarRange()
        otf = self.GetProperty().GetScalarOpacity()
        otf.RemoveAllPoints()
        self._alpha = alpha

        if utils.isSequence(alpha):
            alpha = np.array(alpha)
            if len(alpha.shape)==1: # user passing a flat list e.g. (0.0, 0.3, 0.9, 1)
                for i, al in enumerate(alpha):
                    xalpha = smin + (smax - smin) * i / (len(alpha) - 1)
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
            elif len(alpha.shape)==2: # user passing [(x0,alpha0), ...]
                otf.AddPoint(smin, alpha[0][1])
                for xalpha, al in alpha:
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
                otf.AddPoint(smax, alpha[-1][1])
            #colors.printc("alpha at", round(xalpha, 1), "\tset to", al)

        else:
            otf.AddPoint(smin, alpha) # constant alpha
            otf.AddPoint(smax, alpha)

        return self

    def alphaUnit(self, u=None):
        """
        Defines light attenuation per unit length. Default is 1.
        The larger the unit length, the further light has to travel to attenuate the same amount.

        E.g., if you set the unit distance to 0, you will get full opacity.
        It means that when light travels 0 distance it's already attenuated a finite amount.
        Thus, any finite distance should attenuate all light.
        The larger you make the unit distance, the more transparent the rendering becomes.
        """
        if u is None:
            return self.GetProperty().GetScalarOpacityUnitDistance()
        else:
            self.GetProperty().SetScalarOpacityUnitDistance(u)
            return self


    def shrink(self, fraction=0.8):
        """Shrink the individual tetrahedra to improve visibility."""
        sf = vtk.vtkShrinkFilter()
        sf.SetInputData(self._ugrid)
        sf.SetShrinkFactor(fraction)
        sf.Update()
        return self._update(sf.GetOutput())


    def threshold(self, name=None, above=None, below=None):
        """
        Threshold the tetrahedral mesh by a cell scalar value.
        Reduce to only tets which satisfy the threshold limits.
        """
        th = vtk.vtkThreshold()
        th.SetInputData(self._ugrid)
        ns = self.getArrayNames()

        if name is None:
            if len(ns['CellData']):
                name=ns['CellData'][0]
                th.SetInputArrayToProcess(0,0,0, 1, name)
            elif len(ns['PointData']):
                name=ns['PointData'][0]
                th.SetInputArrayToProcess(0,0,0, 0, name)
            if name is None:
                printc("threshold(): Cannot find active array. Skip.", c=1)
                return self

        if above is not None and below is not None:
            if above<below:
                th.ThresholdBetween(above, below)
            elif above==below:
                return self
            #else:
            #    th.SetInvert(True)
            #    th.ThresholdBetween(below, above)

        elif above is not None:
            th.ThresholdByUpper(above)

        elif below is not None:
            th.ThresholdByLower(below)

        th.Update()
        ugrid = th.GetOutput()
        return self._update(ugrid)


    def cutWithPlane(self, origin=(0,0,0), normal=(1,0,0)):
        """
        Cut the mesh with the plane defined by a point and a normal.

        :param origin: the cutting plane goes through this point
        :param normal: normal of the cutting plane
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
        clipper = vtk.vtkClipDataSet()
        clipper.SetInputData(self._ugrid)
        clipper.SetClipFunction(plane)
        clipper.GenerateClipScalarsOff()
        clipper.GenerateClippedOutputOff()
        clipper.SetValue(0)
        clipper.Update()
        cout = clipper.GetOutput()
        return self._update(cout)


    def cutWithBox(self, box):
        """
        Cut the mesh with the plane defined by a point and a normal.

        Parameter box has format [xmin, xmax, ymin, ymax, zmin, zmax].
        If a Mesh is passed, its bounding box is used.

        Example:

            .. code-block:: python

                from vtkplotter import *
                tetmesh = TetMesh(datadir+'limb_ugrid.vtk')
                tetmesh.color('rainbow')
                cu = Cube(side=500).x(500) # any Mesh works
                tetmesh.cutWithBox(cu).show(axes=1)
        """
        bc = vtk.vtkBoxClipDataSet()
        bc.SetInputData(self._ugrid)
        if isinstance(box, (Mesh, TetMesh)):
            box = box.GetBounds()
        bc.SetBoxClip(*box)
        bc.Update()
        cout = bc.GetOutput()
        return self._update(cout)


    def cutWithMesh(self, mesh, invert=False, onlyTets=False, onlyBoundary=False):
        """
        Cut a ``TetMesh`` mesh with a ``Mesh``.

        :param bool invert: if True return cut off part of the input TetMesh.
        """
        polymesh = mesh.polydata()
        ug = self._ugrid

        scalname = ug.GetCellData().GetScalars().GetName()

        ippd = vtk.vtkImplicitPolyDataDistance()
        ippd.SetInput(polymesh)

        if onlyTets or onlyBoundary:
            clipper = vtk.vtkExtractGeometry()
            clipper.SetInputData(ug)
            clipper.SetImplicitFunction(ippd)
            clipper.SetExtractInside(not invert)
            clipper.SetExtractBoundaryCells(False)
            if onlyBoundary:
                clipper.SetExtractBoundaryCells(True)
                clipper.SetExtractOnlyBoundaryCells(True)
        else:
            signedDistances = vtk.vtkFloatArray()
            signedDistances.SetNumberOfComponents(1)
            signedDistances.SetName("SignedDistances")
            for pointId in range(ug.GetNumberOfPoints()):
                p = ug.GetPoint(pointId)
                signedDistance = ippd.EvaluateFunction(p)
                signedDistances.InsertNextValue(signedDistance)
            ug.GetPointData().SetScalars(signedDistances)
            clipper = vtk.vtkClipDataSet()
            clipper.SetInputData(ug)
            clipper.SetInsideOut(not invert)
            clipper.SetValue(0.0)

        clipper.Update()
        cug = clipper.GetOutput()

        if scalname: # not working
            if self.useCells:
                self.selectCellArray(scalname)
            else:
                self.selectPointArray(scalname)
        self._update(cug)
        return self


    def decimate(self, scalarsName, fraction=0.5, N=None):
        """
        Downsample the number of tets in a TetMesh to a specified fraction.

        :param float fraction: the desired final fraction of the total.
        :param int N: the desired number of final tets

        .. note:: Setting ``fraction=0.1`` leaves 10% of the original nr of tets.
        """
        decimate = vtk.vtkUnstructuredGridQuadricDecimation()
        decimate.SetInputData(self._ugrid)
        decimate.SetScalarsName(scalarsName)

        if N:  # N = desired number of points
            decimate.SetNumberOfTetsOutput(N)
        else:
            decimate.SetTargetReduction(1-fraction)
        decimate.Update()
        return self._update(decimate.GetOutput())


    def subdvide(self):
        """Increase the number of tets of a TetMesh.
        Subdivide one tetrahedron into twelve for every tetra."""
        sd = vtk.vtkSubdivideTetra()
        sd.SetInputData(self._ugrid)
        sd.Update()
        return self._update(sd.GetOutput())


    def extractCellsByID(self, idlist, usePointIDs=False):
        """Return a new TetMesh composed of the specified subset of indices."""
        selectionNode = vtk.vtkSelectionNode()
        if usePointIDs:
            selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
            contcells = vtk.vtkSelectionNode.CONTAINING_CELLS()
            selectionNode.GetProperties().Set(contcells, 1)
        else:
            selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        vidlist = numpy_to_vtkIdTypeArray(np.array(idlist).astype(np.int64))
        selectionNode.SetSelectionList(vidlist)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        es = vtk.vtkExtractSelection()
        es.SetInputData(0, self._ugrid)
        es.SetInputData(1, selection)
        es.Update()
        tm_sel = TetMesh(es.GetOutput())
        pr = vtk.vtkVolumeProperty()
        pr.DeepCopy(self.GetProperty())
        tm_sel.SetProperty(pr)

        #assign the same transformation to the copy
        tm_sel.SetOrigin(self.GetOrigin())
        tm_sel.SetScale(self.GetScale())
        tm_sel.SetOrientation(self.GetOrientation())
        tm_sel.SetPosition(self.GetPosition())
        tm_sel._mapper.SetLookupTable(utils.ctf2lut(self))
        return tm_sel


    def isosurface(self, threshold=True):
        """Return a ``Mesh`` isosurface.

        :param float,list threshold: value or list of values to draw the isosurface(s)
        """
        if not self._ugrid.GetPointData().GetScalars():
            self.mapCellsToPoints()
        scrange = self._ugrid.GetPointData().GetScalars().GetRange()
        cf = vtk.vtkContourFilter() #vtk.vtkContourGrid()
        cf.SetInputData(self._ugrid)

        if utils.isSequence(threshold):
            cf.SetNumberOfContours(len(threshold))
            for i, t in enumerate(threshold):
                cf.SetValue(i, t)
            cf.Update()
        else:
            if threshold is True:
                threshold = (2 * scrange[0] + scrange[1]) / 3.0
                #print('automatic threshold set to ' + utils.precision(threshold, 3), end=' ')
                #print('in [' + utils.precision(scrange[0], 3) + ', ' + utils.precision(scrange[1], 3)+']')
            cf.SetValue(0, threshold)
            cf.Update()

        clp = vtk.vtkCleanPolyData()
        clp.SetInputData(cf.GetOutput())
        clp.Update()
        msh = Mesh(clp.GetOutput(), c=None).phong()
        msh._mapper.SetLookupTable(utils.ctf2lut(self))
        return msh


    def slice(self, origin=(0,0,0), normal=(1,0,0)):
        """Return a 2D slice of the mesh by a plane passing through origin and
        assigned normal."""
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
        cc.SetInputData(self._ugrid)
        cc.SetCutFunction(plane)
        cc.Update()
        msh = Mesh(cc.GetOutput()).flat().lighting('ambient')
        msh._mapper.SetLookupTable(utils.ctf2lut(self))
        return msh



###################################################################################
# def extractCellsByType(obj, types=(7,)):    ### VTK9 only
#     """Extract cells of a specified type.
#     Given an input vtkDataSet and a list of cell types, produce an output
#     containing only cells of the specified type(s).
#     Find `here <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>`_
#     the list of possible cell types.
#     """
#     ef = vtk.vtkExtractCellsByType()
#     for ct in types:
#         ef.AddCellType(ct)
#     ef.Update()
#     return Mesh(ef.GetOutput())

# def _extractTets(ugrid):
#     uarr = ugrid.GetCellTypesArray()
#     ctarr10 = np.where(vtk_to_numpy(uarr)==10)[0]
#     uarr10 = numpy_to_vtkIdTypeArray(ctarr10, deep=False)
#     selectionNode = vtk.vtkSelectionNode()
#     selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
#     selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
#     selectionNode.SetSelectionList(uarr10)
#     selection = vtk.vtkSelection()
#     selection.AddNode(selectionNode)
#     es = vtk.vtkExtractSelection()
#     es.SetInputData(0, ugrid)
#     es.SetInputData(1, selection)
#     es.Update()
#     ugrid_tets = es.GetOutput()
#     return ugrid_tets


























