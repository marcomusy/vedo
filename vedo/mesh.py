from __future__ import division, print_function
import numpy as np
import os
import vtk
import vedo.colors as colors
import vedo.docs as docs
import vedo.settings as settings
import vedo.utils as utils
from vedo.pointcloud import Points

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

__doc__ = ("""Submodule to manage polygonal meshes."""
    + docs._defs
)

__all__ = ["Mesh",
           "merge",
           ]


####################################################
def merge(*meshs):
    """
    Build a new mesh formed by the fusion of the polygonal meshes of the input objects.
    Similar to Assembly, but in this case the input objects become a single mesh.

    .. hint:: |thinplate_grid.py|_ |value-iteration.py|_

        |thinplate_grid| |value-iteration|
    """
    acts = []
    for a in utils.flatten(meshs):
        if a:
            acts += [a]

    if len(acts) == 1:
        return acts[0].clone()
    elif len(acts) == 0:
        return None

    polyapp = vtk.vtkAppendPolyData()
    for a in acts:
        if isinstance(a, vtk.vtkPolyData):
            polyapp.AddInputData(a)
        else:
            polyapp.AddInputData(a.polydata())
    polyapp.Update()
    msh = Mesh(polyapp.GetOutput())
    if isinstance(acts[0], vtk.vtkActor):
        cprp = vtk.vtkProperty()
        cprp.DeepCopy(acts[0].GetProperty())
        msh.SetProperty(cprp)
    return msh


####################################################
class Mesh(Points):
    """
    Build an instance of object ``Mesh`` derived from ``PointCloud``.

    Input can be ``vtkPolyData``, ``vtkActor``, or a python list of [vertices, faces].

    If input is any of ``vtkUnstructuredGrid``, ``vtkStructuredGrid`` or ``vtkRectilinearGrid``
    the geometry is extracted.
    In this case the original VTK data structures can be accessed with: ``mesh.inputdata()``.

    Finally input can be a list of vertices and their connectivity (faces of the polygonal mesh).
    For point clouds - e.i. no faces - just substitute the `faces` list with ``None``.

    E.g.: `Mesh( [ [[x1,y1,z1],[x2,y2,z2], ...],  [[0,1,2], [1,2,3], ...] ] )`

    :param c: color in RGB format, hex, symbol or name
    :param float alpha: opacity value
    :param bool wire:  show surface as wireframe
    :param bc: backface color of internal surface
    :param str texture: jpg file name or surface texture name
    :param bool computeNormals: compute point and cell normals at creation

    .. hint:: A mesh can be built from vertices and their connectivity. See e.g.:

        |buildmesh| |buildmesh.py|_
    """

    def __init__(
        self,
        inputobj=None,
        c=None,
        alpha=1,
        computeNormals=False,
    ):
        Points.__init__(self)

        self._mapper.SetInterpolateScalarsBeforeMapping(settings.interpolateScalarsBeforeMapping)

        if settings.usePolygonOffset:
            self._mapper.SetResolveCoincidentTopologyToPolygonOffset()
            pof, pou = settings.polygonOffsetFactor, settings.polygonOffsetUnits
            self._mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(pof, pou)

        inputtype = str(type(inputobj))

        if inputobj is None:
            pass

        elif isinstance(inputobj, Mesh) or isinstance(inputobj, vtk.vtkActor):
            polyCopy = vtk.vtkPolyData()
            polyCopy.DeepCopy(inputobj.GetMapper().GetInput())
            self._polydata = polyCopy
            self._mapper.SetInputData(polyCopy)
            self._mapper.SetScalarVisibility(inputobj.GetMapper().GetScalarVisibility())
            pr = vtk.vtkProperty()
            pr.DeepCopy(inputobj.GetProperty())
            self.SetProperty(pr)

        elif isinstance(inputobj, vtk.vtkPolyData):
            if inputobj.GetNumberOfCells() == 0:
                carr = vtk.vtkCellArray()
                for i in range(inputobj.GetNumberOfPoints()):
                    carr.InsertNextCell(1)
                    carr.InsertCellPoint(i)
                inputobj.SetVerts(carr)
            self._polydata = inputobj  # cache vtkPolyData and mapper for speed

        elif isinstance(inputobj, (vtk.vtkStructuredGrid, vtk.vtkRectilinearGrid)):
            if settings.visibleGridEdges:
                gf = vtk.vtkExtractEdges()
                gf.SetInputData(inputobj)
            else:
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(inputobj)
            gf.Update()
            self._polydata = gf.GetOutput()

        elif "trimesh" in inputtype:
            tact = utils.trimesh2vedo(inputobj, alphaPerCell=False)
            self._polydata = tact.polydata()

        elif "meshio" in inputtype: # meshio-4.0.11
            if len(inputobj.cells):
                mcells =[]
                for cellblock in inputobj.cells:
                    if cellblock.type in ("triangle", "quad"):
                        mcells += cellblock.data.tolist()
                self._polydata = utils.buildPolyData(inputobj.points, mcells)
            else:
                self._polydata = utils.buildPolyData(inputobj.points, None)
            # add arrays:
            try:
                if len(inputobj.point_data):
                    for k in inputobj.point_data.keys():
                        vdata = numpy_to_vtk(inputobj.point_data[k], deep=True)
                        vdata.SetName(str(k))
                        self._polydata.GetPointData().AddArray(vdata)
            except AssertionError:
                print("Could not add meshio point data, skip.")
            try:
                if len(inputobj.cell_data):
                    for k in inputobj.cell_data.keys():
                        vdata = numpy_to_vtk(inputobj.cell_data[k], deep=True)
                        vdata.SetName(str(k))
                        self._polydata.GetCellData().AddArray(vdata)
            except AssertionError:
                print("Could not add meshio cell data, skip.")

        elif utils.isSequence(inputobj):
            ninp = len(inputobj)
            if ninp == 0:
                self._polydata = vtk.vtkPolyData()
            elif ninp == 2: # assume [vertices, faces]
                self._polydata = utils.buildPolyData(inputobj[0], inputobj[1])
            else:           # assume [vertices] or vertices
                self._polydata = utils.buildPolyData(inputobj, None)

        elif hasattr(inputobj, "GetOutput"): # passing vtk object
            if hasattr(inputobj, "Update"): inputobj.Update()
            if isinstance(inputobj.GetOutput(), vtk.vtkPolyData):
                self._polydata = inputobj.GetOutput()
            else:
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(inputobj.GetOutput())
                gf.Update()
                self._polydata = gf.GetOutput()

        elif isinstance(inputobj, str):
            from vedo.io import load
            dataset = load(inputobj)
            if "TetMesh" in str(type(dataset)):
                self._polydata = dataset.tomesh().polydata()
            else:
                self._polydata = dataset.polydata()

        else:
            colors.printc("Error: cannot build mesh from type:\n", inputtype, c='r')
            raise RuntimeError()


        if settings.computeNormals is not None:
            computeNormals = settings.computeNormals

        if self._polydata:
            if computeNormals:
                pdnorm = vtk.vtkPolyDataNormals()
                pdnorm.SetInputData(self._polydata)
                pdnorm.ComputePointNormalsOn()
                pdnorm.ComputeCellNormalsOn()
                pdnorm.FlipNormalsOff()
                pdnorm.ConsistencyOn()
                pdnorm.Update()
                self._polydata = pdnorm.GetOutput()

        self._mapper.SetInputData(self._polydata)

        self.line_locator = None

        self._bfprop = None  # backface property holder
        self._scals_idx = 0  # index of the active scalar changed from CLI
        self._ligthingnr = 0

        prp = self.GetProperty()
        prp.SetInterpolationToPhong()

        if settings.renderLinesAsTubes:
            if hasattr(prp, 'RenderLinesAsTubesOn'):
                prp.RenderLinesAsTubesOn()

        # set the color by c or by scalar
        if self._polydata:

            arrexists = False

            if c is None:
                ptdata = self._polydata.GetPointData()
                cldata = self._polydata.GetCellData()
                exclude = ['normals', 'tcoord']

                if cldata.GetNumberOfArrays():
                    for i in range(cldata.GetNumberOfArrays()):
                        iarr = cldata.GetArray(i)
                        if iarr:
                            icname = iarr.GetName()
                            if icname and all(s not in icname.lower() for s in exclude):
                                cldata.SetActiveScalars(icname)
                                self._mapper.ScalarVisibilityOn()
                                self._mapper.SetScalarModeToUseCellData()
                                self._mapper.SetScalarRange(iarr.GetRange())
                                arrexists = True
                                break # stop at first good one

                # point come after so it has priority
                if ptdata.GetNumberOfArrays():
                    for i in range(ptdata.GetNumberOfArrays()):
                        iarr = ptdata.GetArray(i)
                        if iarr:
                            ipname = iarr.GetName()
                            if ipname and all(s not in ipname.lower() for s in exclude):
                                ptdata.SetActiveScalars(ipname)
                                self._mapper.ScalarVisibilityOn()
                                self._mapper.SetScalarModeToUsePointData()
                                self._mapper.SetScalarRange(iarr.GetRange())
                                arrexists = True
                                break # stop at first good one

            if not arrexists:
                if c is None:
                    c = "gold"
                c = colors.getColor(c)
                prp.SetColor(c)
                prp.SetAmbient(0.1)
                prp.SetDiffuse(1)
                prp.SetSpecular(.05)
                prp.SetSpecularPower(5)
                self._mapper.ScalarVisibilityOff()

        if alpha is not None:
            prp.SetOpacity(alpha)


    ###############################################
    def clone(self):
        """
        Clone a ``Mesh`` object to make an exact copy of it.

        |mirror| |mirror.py|_
        """
        poly = self.polydata(False)
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(poly)

        cloned = Mesh(polyCopy)
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.GetProperty())
        cloned.SetProperty(pr)

        if self.GetBackfaceProperty():
            bfpr = vtk.vtkProperty()
            bfpr.DeepCopy(self.GetBackfaceProperty())
            cloned.SetBackfaceProperty(bfpr)

        # assign the same transformation to the copy
        cloned.SetOrigin(self.GetOrigin())
        cloned.SetScale(self.GetScale())
        cloned.SetOrientation(self.GetOrientation())
        cloned.SetPosition(self.GetPosition())

        cloned._mapper.SetScalarVisibility(self._mapper.GetScalarVisibility())
        cloned._mapper.SetScalarRange(self._mapper.GetScalarRange())
        cloned._mapper.SetColorMode(self._mapper.GetColorMode())
        lsr = self._mapper.GetUseLookupTableScalarRange()
        cloned._mapper.SetUseLookupTableScalarRange(lsr)
        cloned._mapper.SetScalarMode(self._mapper.GetScalarMode())
        lut = self._mapper.GetLookupTable()
        if lut:
            cloned._mapper.SetLookupTable(lut)

        cloned.base = self.base
        cloned.top = self.top
        cloned.name = self.name
        if self.trail:
            n = len(self.trailPoints)
            cloned.addTrail(self.trailOffset, self.trailSegmentSize*n, n,
                            None, None, self.trail.GetProperty().GetLineWidth())
        if self.shadow:
            cloned.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                             self.shadow.GetProperty().GetColor(),
                             self.shadow.GetProperty().GetOpacity())
        return cloned


    def faces(self):
        """Get cell polygonal connectivity ids as a python ``list``.
        The output format is: [[id0 ... idn], [id0 ... idm],  etc].
        """
        arr1d = vtk_to_numpy(self._polydata.GetPolys().GetData())
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
        return conn # cannot always make a numpy array of it!


    def lines(self, flat=False):
        """Get lines connectivity ids as a numpy array.
        Default format is [[id0,id1], [id3,id4], ...]

        :param bool flat: 1D numpy array as [2, 10,20, 3, 10,11,12, 2, 70,80, ...]
        """
        #Get cell connettivity ids as a 1D array. The vtk format is:
        #    [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        arr1d = vtk_to_numpy(self.polydata(False).GetLines().GetData())

        if arr1d is None:
            return []

        if flat:
            return arr1d

        i = 0
        conn = []
        n = len(arr1d)
        for idummy in range(n):
            cell = [arr1d[i+k+1] for k in range(arr1d[i])]
            conn.append(cell)
            i += arr1d[i]+1
            if i >= n:
                break

        return conn # cannot always make a numpy array of it!

    def texture(self, tname,
                tcoords=None,
                interpolate=True,
                repeat=True,
                edgeClamp=False,
                scale=None,
                ushift=None,
                vshift=None,
                ):
        """Assign a texture to mesh from image file or predefined texture `tname`.
        If tname is ``None`` texture is disabled.
        Input tname can also be an array of shape (n,m,3).

        :param bool interpolate: turn on/off linear interpolation of the texture map when rendering.
        :param bool repeat: repeat of the texture when tcoords extend beyond the [0,1] range.
        :param bool edgeClamp: turn on/off the clamping of the texture map when
            the texture coords extend beyond the [0,1] range.
            Only used when repeat is False, and edge clamping is supported by the graphics card.
        :param bool scale: scale the texture image by this factor
        :param bool ushift: shift u-coordinates of texture by this amaount
        :param bool vshift: shift v-coordinates of texture by this amaount
        """
        pd = self.polydata(False)
        if tname is None:
            pd.GetPointData().SetTCoords(None)
            pd.GetPointData().Modified()
            return self

        if utils.isSequence(tname):
            from PIL import Image
            from tempfile import NamedTemporaryFile
            tmp_file = NamedTemporaryFile()
            im = Image.fromarray(tname)
            im.save(tmp_file.name+".bmp")
            tname = tmp_file.name+".bmp"

        if isinstance(tname, vtk.vtkTexture):
            tu = tname
        else:
            if tcoords is not None:
                if not isinstance(tcoords, np.ndarray):
                    tcoords = np.array(tcoords)
                if tcoords.ndim != 2:
                    colors.printc('tcoords must be a 2-dimensional array', c='r')
                    return self
                if tcoords.shape[0] != pd.GetNumberOfPoints():
                    colors.printc('Error in texture(): nr of texture coords must match nr of points', c='r')
                    return self
                if tcoords.shape[1] != 2:
                    colors.printc('Error in texture(): vector must have 2 components', c='r')
                tarr = numpy_to_vtk(np.ascontiguousarray(tcoords), deep=True)
                tarr.SetName('TCoordinates')
                pd.GetPointData().SetTCoords(tarr)
                pd.GetPointData().Modified()
            else:
                if not pd.GetPointData().GetTCoords():
                    tmapper = vtk.vtkTextureMapToPlane()
                    tmapper.AutomaticPlaneGenerationOn()
                    tmapper.SetInputData(pd)
                    tmapper.Update()
                    tc = tmapper.GetOutput().GetPointData().GetTCoords()
                    if scale or ushift or vshift:
                        ntc = vtk_to_numpy(tc)
                        if scale: ntc *= scale
                        if ushift: ntc[:,0] += ushift
                        if vshift: ntc[:,1] += vshift
                        tc = numpy_to_vtk(tc, deep=True)
                    pd.GetPointData().SetTCoords(tc)
                    pd.GetPointData().Modified()

            fn = settings.textures_path + tname + ".jpg"
            if os.path.exists(tname):
                fn = tname
            elif not os.path.exists(fn):
                colors.printc("File does not exist or texture", tname,
                              "not found in", settings.textures_path, c="r")
                colors.printc("\pin Available built-in textures:", c="m", end=" ")
                for ff in os.listdir(settings.textures_path):
                    colors.printc(ff.split(".")[0], end=" ", c="m")
                print()
                return self

            fnl = fn.lower()
            if ".jpg" in fnl or ".jpeg" in fnl:
                reader = vtk.vtkJPEGReader()
            elif ".png" in fnl:
                reader = vtk.vtkPNGReader()
            elif ".bmp" in fnl:
                reader = vtk.vtkBMPReader()
            else:
                colors.printc("Error in texture(): supported files, PNG, BMP or JPG", c="r")
                return self
            reader.SetFileName(fn)
            reader.Update()

            tu = vtk.vtkTexture()
            tu.SetInputData(reader.GetOutput())
            tu.SetInterpolate(interpolate)
            tu.SetRepeat(repeat)
            tu.SetEdgeClamp(edgeClamp)

        self.GetProperty().SetColor(1, 1, 1)
        self._mapper.ScalarVisibilityOff()
        self.SetTexture(tu)
        self.Modified()
        return self



    def computeNormals(self, points=True, cells=True):
        """Compute cell and vertex normals for the mesh.

        .. warning:: Mesh gets modified, output can have a different nr. of vertices.
        """
        poly = self.polydata(False)
        pdnorm = vtk.vtkPolyDataNormals()
        pdnorm.SetInputData(poly)
        pdnorm.SetComputePointNormals(points)
        pdnorm.SetComputeCellNormals(cells)
        pdnorm.FlipNormalsOff()
        pdnorm.ConsistencyOn()
        pdnorm.Update()
        return self._update(pdnorm.GetOutput())



    def reverse(self, cells=True, normals=False):
        """
        Reverse the order of polygonal cells
        and/or reverse the direction of point and cell normals.
        Two flags are used to control these operations:

        - `cells=True` reverses the order of the indices in the cell connectivity list.

        - `normals=True` reverses the normals by multiplying the normal vector by -1
            (both point and cell normals, if present).
        """
        poly = self.polydata(False)
        rev = vtk.vtkReverseSense()
        if cells:
            rev.ReverseCellsOn()
        else:
            rev.ReverseCellsOff()
        if normals:
            rev.ReverseNormalsOn()
        else:
            rev.ReverseNormalsOff()
        rev.SetInputData(poly)
        rev.Update()
        return self._update(rev.GetOutput())

    def wireframe(self, value=True):
        """Set mesh's representation as wireframe or solid surface.
        Same as `mesh.wireframe()`."""
        if value:
            self.GetProperty().SetRepresentationToWireframe()
        else:
            self.GetProperty().SetRepresentationToSurface()
        return self

    def flat(self):
        """Set surface interpolation to Flat.

        |wikiphong|
        """
        self.GetProperty().SetInterpolationToFlat()
        return self

    def phong(self):
        """Set surface interpolation to Phong."""
        self.GetProperty().SetInterpolationToPhong()
        return self

    def backFaceCulling(self, value=True):
        """Set culling of polygons based on orientation
        of normal with respect to camera."""
        self.GetProperty().SetBackfaceCulling(value)
        return self

    def frontFaceCulling(self, value=True):
        """Set culling of polygons based on orientation of normal with respect to camera."""
        self.GetProperty().SetFrontfaceCulling(value)
        return self

    def backColor(self, bc=None):
        """
        Set/get mesh's backface color.
        """
        backProp = self.GetBackfaceProperty()

        if bc is None:
            if backProp:
                return backProp.GetDiffuseColor()
            return self

        if self.GetProperty().GetOpacity() < 1:
            #colors.printc("In backColor(): only active for alpha=1", c="y")
            return self

        if not backProp:
            backProp = vtk.vtkProperty()

        backProp.SetDiffuseColor(colors.getColor(bc))
        backProp.SetOpacity(self.GetProperty().GetOpacity())
        self.SetBackfaceProperty(backProp)
        self._mapper.ScalarVisibilityOff()
        return self

    def bc(self, backColor=False):
        """Shortcut for `mesh.backColor()`. """
        return self.backColor(backColor)

    def lineWidth(self, lw=None):
        """Set/get width of mesh edges. Same as `lw()`."""
        if lw is not None:
            if lw == 0:
                self.GetProperty().EdgeVisibilityOff()
                self.GetProperty().SetRepresentationToSurface()
                return self
            self.GetProperty().EdgeVisibilityOn()
            self.GetProperty().SetLineWidth(lw)
        else:
            return self.GetProperty().GetLineWidth()
        return self

    def lw(self, lineWidth=None):
        """Set/get width of mesh edges. Same as `lineWidth()`."""
        return self.lineWidth(lineWidth)

    def lineColor(self, lc=None):
        """Set/get color of mesh edges. Same as `lc()`."""
        if lc is not None:
            if "ireframe" in self.GetProperty().GetRepresentationAsString():
                self.GetProperty().EdgeVisibilityOff()
                self.color(lc)
                return self
            self.GetProperty().EdgeVisibilityOn()
            self.GetProperty().SetEdgeColor(colors.getColor(lc))
        else:
            return self.GetProperty().GetEdgeColor()
        return self

    def lc(self, lineColor=None):
        """Set/get color of mesh edges. Same as `lineColor()`."""
        return self.lineColor(lineColor)

    def quality(self, measure=6, cmap='RdYlBu'):
        """
        Calculate functions of quality of the elements of a triangular mesh.
        See class `vtkMeshQuality <https://vtk.org/doc/nightly/html/classvtkMeshQuality.html>`_
        for explanation.

        :param int measure: type of estimator

            - EDGE RATIO, 0
            - ASPECT RATIO, 1
            - RADIUS RATIO, 2
            - ASPECT FROBENIUS, 3
            - MED ASPECT_FROBENIUS, 4
            - MAX ASPECT FROBENIUS, 5
            - MIN_ANGLE, 6
            - COLLAPSE RATIO, 7
            - MAX ANGLE, 8
            - CONDITION, 9
            - SCALED JACOBIAN, 10
            - SHEAR, 11
            - RELATIVE SIZE SQUARED, 12
            - SHAPE, 13
            - SHAPE AND SIZE, 14
            - DISTORTION, 15
            - MAX EDGE RATIO, 16
            - SKEW, 17
            - TAPER, 18
            - VOLUME, 19
            - STRETCH, 20
            - DIAGONAL, 21
            - DIMENSION, 22
            - ODDY, 23
            - SHEAR AND SIZE, 24
            - JACOBIAN, 25
            - WARPAGE, 26
            - ASPECT GAMMA, 27
            - AREA, 28
            - ASPECT BETA, 29

        |meshquality| |meshquality.py|_
        """
        qf = vtk.vtkMeshQuality()
        qf.SetInputData(self.polydata(False))
        qf.SetTriangleQualityMeasure(measure)
        qf.SaveCellQualityOn()
        qf.Update()
        pd = qf.GetOutput()
        varr = pd.GetCellData().GetArray('Quality')
        self.addCellArray(varr, "Quality")
        self.cmap(cmap, "Quality", on='cells')
        arr = vtk_to_numpy(pd.GetCellData().GetArray('Quality'))
        return arr


    def volume(self, value=None):
        """Get/set the volume occupied by mesh."""
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata(False))
        mass.Update()
        v = mass.GetVolume()
        if value is not None:
            if not v:
                colors.printc("Volume is zero: cannot rescale.", c='r', end="")
                colors.printc(" Consider adding mesh.triangulate()", c='r')
                return self
            self.scale(value / v)
            return self
        else:
            return v

    def area(self, value=None):
        """Get/set the surface area of mesh.

        .. hint:: |largestregion.py|_
        """
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata(False))
        mass.Update()
        ar = mass.GetSurfaceArea()
        if value is not None:
            if not ar:
                colors.printc("Area is zero: cannot rescale.", c='r', end="")
                colors.printc(" Consider adding mesh.triangulate()", c='r')
                return self
            self.scale(value / ar)
            return self
        else:
            return ar


    def distanceToMesh(self, mesh, signed=False, negate=False):
        '''
        Computes the (signed) distance from one mesh to another.

        |distance2mesh| |distance2mesh.py|_
        '''
        poly1 = self.polydata()
        poly2 = mesh.polydata()
        df = vtk.vtkDistancePolyDataFilter()
        df.ComputeSecondDistanceOff()
        df.SetInputData(0, poly1)
        df.SetInputData(1, poly2)
        if signed:
            df.SignedDistanceOn()
        else:
            df.SignedDistanceOff()
        if negate:
            df.NegateDistanceOn()
        df.Update()

        scals = df.GetOutput().GetPointData().GetScalars()
        poly1.GetPointData().AddArray(scals)

        poly1.GetPointData().SetActiveScalars(scals.GetName())
        rng = scals.GetRange()
        self._mapper.SetScalarRange(rng[0], rng[1])
        self._mapper.ScalarVisibilityOn()
        return self


    def shrink(self, fraction=0.85):
        """Shrink the triangle polydata in the representation of the input mesh.

        Example:
            .. code-block:: python

                from vedo import *
                pot = load(datadir + 'teapot.vtk').shrink(0.75)
                s = Sphere(r=0.2).pos(0,0,-0.5)
                show(pot, s)

            |shrink| |shrink.py|_
        """
        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputData(self._polydata)
        shrink.SetShrinkFactor(fraction)
        shrink.Update()
        return self._update(shrink.GetOutput())


    def stretch(self, q1, q2):
        """Stretch mesh between points `q1` and `q2`. Mesh is not affected.

        |aspring| |aspring.py|_

        .. note:: for ``Mesh`` objects like helices, Line, cylinders, cones etc.,
            two attributes ``mesh.base``, and ``mesh.top`` are already defined.
        """
        if self.base is None:
            colors.printc('Error in stretch(): Please define vectors', c='r')
            colors.printc('   mesh.base and mesh.top at creation.', c='r')
            raise RuntimeError()

        p1, p2 = self.base, self.top
        q1, q2, z = np.array(q1), np.array(q2), np.array([0, 0, 1])
        plength = np.linalg.norm(p2 - p1)
        qlength = np.linalg.norm(q2 - q1)
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.Translate(-p1)
        cosa = np.dot(p2 - p1, z) / plength
        n = np.cross(p2 - p1, z)
        T.RotateWXYZ(np.rad2deg(np.arccos(cosa)), n)
        T.Scale(1, 1, qlength / plength)

        cosa = np.dot(q2 - q1, z) / qlength
        n = np.cross(q2 - q1, z)
        T.RotateWXYZ(-np.rad2deg(np.arccos(cosa)), n)
        T.Translate(q1)

        self.SetUserMatrix(T.GetMatrix())
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self

    def crop(self,
             top=None, bottom=None, right=None, left=None, front=None, back=None,
             bounds=None,
        ):
        """Crop an ``Mesh`` object.

        :param float top:    fraction to crop from the top plane (positive z)
        :param float bottom: fraction to crop from the bottom plane (negative z)
        :param float front:  fraction to crop from the front plane (positive y)
        :param float back:   fraction to crop from the back plane (negative y)
        :param float right:  fraction to crop from the right plane (positive x)
        :param float left:   fraction to crop from the left plane (negative x)
        :param list bounds:  direct list of bounds passed as [x0,x1, y0,y1, z0,z1]

        Example:
            .. code-block:: python

                from vedo import Sphere
                Sphere().crop(right=0.3, left=0.1).show()

            |cropped|
        """
        cu = vtk.vtkBox()
        x0, x1, y0, y1, z0, z1 = self.GetBounds()
        pos = np.array(self.GetPosition())
        x0, y0, z0 = [x0, y0, z0] - pos
        x1, y1, z1 = [x1, y1, z1] - pos

        if bounds is None:
            dx, dy, dz = x1-x0, y1-y0, z1-z0
            if top:    z1 = z1 - top*dz
            if bottom: z0 = z0 + bottom*dz
            if front:  y1 = y1 - front*dy
            if back:   y0 = y0 + back*dy
            if right:  x1 = x1 - right*dx
            if left:   x0 = x0 + left*dx
            bounds = (x0, x1, y0, y1, z0, z1)
        else:
            if bounds[0] is None: bounds[0]=x0
            if bounds[1] is None: bounds[1]=x1
            if bounds[2] is None: bounds[2]=y0
            if bounds[3] is None: bounds[3]=y1
            if bounds[4] is None: bounds[4]=z0
            if bounds[5] is None: bounds[5]=z1
        cu.SetBounds(bounds)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self._polydata)
        clipper.SetClipFunction(cu)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())
        return self

    def cutWithPlane(self, origin=(0, 0, 0), normal=(1, 0, 0), returnCut=False):
        """
        Cut the mesh with the plane defined by a point and a normal.

        :param origin: the cutting plane goes through this point
        :param normal: normal of the cutting plane

        :Example:
            .. code-block:: python

                from vedo import Cube

                cube = Cube().cutWithPlane(normal=(1,1,1))
                cube.bc('pink').show()

            |cutcube|

        |trail| |trail.py|_
        """
        s = str(normal)
        if "x" in s:
            normal = (1, 0, 0)
            if '-' in s: normal = -np.array(normal)
        elif "y" in s:
            normal = (0, 1, 0)
            if '-' in s: normal = -np.array(normal)
        elif "z" in s:
            normal = (0, 0, 1)
            if '-' in s: normal = -np.array(normal)
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.polydata(True)) # must be True
        clipper.SetClipFunction(plane)
        if returnCut:
            clipper.GenerateClippedOutputOn()
        else:
            clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()

        cpoly = clipper.GetOutput()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _polydata is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(clipper.GetOutput())
            tf.Update()
            self._update(tf.GetOutput())

        if returnCut:
            c = self.GetProperty().GetColor()
            xpoly = clipper.GetClippedOutput()
            if self.GetIsIdentity():
                return Mesh(xpoly, c, 0.1).wireframe(True)
            else:
                tfx = vtk.vtkTransformPolyDataFilter()
                tfx.SetTransform(tr)
                tfx.SetInputData(xpoly)
                tfx.Update()
                tfxpoly = tfx.GetOutput()
                restmesh = Mesh(tfxpoly, c, 0.1).wireframe(True)
                restmesh.SetScale(self.GetScale())
                restmesh.SetOrientation(self.GetOrientation())
                restmesh.SetPosition(self.GetPosition())
                return restmesh
        else:
            return self


    def cutWithMesh(self, mesh, invert=False):
        """
        Cut an ``Mesh`` mesh with another ``Mesh``.

        :param bool invert: if True return cut off part of Mesh.

        .. hint:: |cutWithMesh.py|_ |cutAndCap.py|_

            |cutWithMesh| |cutAndCap|
        """
        polymesh = mesh.polydata()
        poly = self.polydata()

        # Create an array to hold distance information
        signedDistances = vtk.vtkFloatArray()
        signedDistances.SetNumberOfComponents(1)
        signedDistances.SetName("SignedDistances")

        # implicit function that will be used to slice the mesh
        ippd = vtk.vtkImplicitPolyDataDistance()
        ippd.SetInput(polymesh)

        # Evaluate the signed distance function at all of the grid points
        for pointId in range(poly.GetNumberOfPoints()):
            p = poly.GetPoint(pointId)
            signedDistance = ippd.EvaluateFunction(p)
            signedDistances.InsertNextValue(signedDistance)

        currentscals = poly.GetPointData().GetScalars()
        if currentscals:
            currentscals = currentscals.GetName()

        poly.GetPointData().AddArray(signedDistances)
        poly.GetPointData().SetActiveScalars("SignedDistances")

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        clipper.SetInsideOut(not invert)
        clipper.SetValue(0.0)
        clipper.Update()
        cpoly = clipper.GetOutput()

        if currentscals:
            cpoly.GetPointData().SetActiveScalars(currentscals)

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _polydata is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(clipper.GetOutput())
            tf.Update()
            self._update(tf.GetOutput())

        return self


    def cutWithPointLoop(self, points, invert=False):
        """
        Cut an ``Mesh`` object with a set of points forming a closed loop.
        """
        if isinstance(points, Points):
            vpts = points.polydata().GetPoints()
            points = points.points()
        else:
            vpts = vtk.vtkPoints()
            for p in points:
                vpts.InsertNextPoint(p)

        spol = vtk.vtkSelectPolyData()
        spol.SetLoop(vpts)
        spol.GenerateSelectionScalarsOn()
        spol.GenerateUnselectedOutputOff()
        spol.SetInputData(self.polydata())
        spol.Update()

        # use vtkClipDataSet to slice the grid with the polydata
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(spol.GetOutput())
        clipper.SetInsideOut(not invert)
        clipper.SetValue(0.0)
        clipper.Update()
        cpoly = clipper.GetOutput()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _polydata is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(clipper.GetOutput())
            tf.Update()
            self._update(tf.GetOutput())
        return self


    def cap(self, returnCap=False):
        """
        Generate a "cap" on a clipped mesh, or caps sharp edges.

        |cutAndCap| |cutAndCap.py|_
        """
        poly = self._polydata

        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(poly)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ManifoldEdgesOff()
        fe.Update()

        stripper = vtk.vtkStripper()
        stripper.SetInputData(fe.GetOutput())
        stripper.Update()

        boundaryPoly = vtk.vtkPolyData()
        boundaryPoly.SetPoints(stripper.GetOutput().GetPoints())
        boundaryPoly.SetPolys(stripper.GetOutput().GetLines())

        tf = vtk.vtkTriangleFilter()
        tf.SetInputData(boundaryPoly)
        tf.Update()

        if returnCap:
            m = Mesh(tf.GetOutput())
            # assign the same transformation to the copy
            m.SetOrigin(self.GetOrigin())
            m.SetScale(self.GetScale())
            m.SetOrientation(self.GetOrientation())
            m.SetPosition(self.GetPosition())
            return m
        else:
            polyapp = vtk.vtkAppendPolyData()
            polyapp.AddInputData(poly)
            polyapp.AddInputData(tf.GetOutput())
            polyapp.Update()
            return self._update(polyapp.GetOutput()).clean().phong()


    def join(self, polys=True):
        """
        Generate triangle strips and/or polylines from
        input polygons, triangle strips, and lines.

        Input polygons are assembled into triangle strips only if they are triangles;
        other types of polygons are passed through to the output and not stripped.
        Use mesh.triangulate() to triangulate non-triangular polygons prior to running
        this filter if you need to strip all the data.

        Also note that if triangle strips or polylines are present in the input
        they are passed through and not joined nor extended.
        If you wish to strip these use mesh.triangulate() to fragment the input
        into triangles and lines prior to applying strip().

        :param bool polys: polygonal segments will be joined if they are contiguous

        :Warning:

            If triangle strips or polylines exist in the input data
            they will be passed through to the output data.
            This filter will only construct triangle strips if triangle polygons
            are available; and will only construct polylines if lines are available.
        """
        sf = vtk.vtkStripper()
        sf.SetPassThroughCellIds(True)
        sf.SetPassThroughPointIds(True)
        sf.SetJoinContiguousSegments(polys)
        sf.SetInputData(self.polydata(False))
        sf.Update()
        return self._update(sf.GetOutput())


    def threshold(self, scalars, above=None, below=None, useCells=False):
        """
        Extracts cells where scalar value satisfies threshold criterion.

        :param str,list scalars: name of the scalars array.
        :param float above: minimum value of the scalar
        :param float below: maximum value of the scalar
        :param bool useCells: if `True`, assume array scalars refers to cells.

        |mesh_threshold| |mesh_threshold.py|_
        """
        if utils.isSequence(scalars):
            if useCells:
                self.addCellArray(scalars, "threshold")
            else:
                self.addPointArray(scalars, "threshold")
            scalars = "threshold"
        else: # string is passed
            if useCells:
                arr = self.getCellArray(scalars)
            else:
                arr = self.getPointArray(scalars)
            if arr is None:
                colors.printc("No scalars found with name/nr:", scalars, c='r')
                colors.printc("Available scalars are:\n", self.getArrayNames(), c='y')
                raise RuntimeError()

        thres = vtk.vtkThreshold()
        thres.SetInputData(self._polydata)

        if useCells:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        thres.SetInputArrayToProcess(0, 0, 0, asso, scalars)

        #        if above is not None and below is not None:
        #            if above<below:
        #                thres.ThresholdBetween(above, below)
        #            elif above==below:
        #                return self
        #            else:
        #                thres.InvertOn()
        #                thres.ThresholdBetween(below, above)
        #        elif above is not None:
        #            thres.ThresholdByUpper(above)
        #        elif below is not None:
        #            thres.ThresholdByLower(below)

        if above is None and below is not None:
            thres.ThresholdByLower(below)
        elif below is None and above is not None:
            thres.ThresholdByUpper(above)
        else:
            thres.ThresholdBetween(above, below)
        thres.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(thres.GetOutput())
        gf.Update()
        return self._update(gf.GetOutput())


    def triangulate(self, verts=True, lines=True):
        """
        Converts mesh polygons into triangles.

        If the input mesh is only made of 2D lines (no faces) the output will be a triangulation
        that fills the internal area. The contours may be concave, and may even contain holes,
        i.e. a contour may contain an internal contour winding in the opposite
        direction to indicate that it is a hole.

        :param bool verts: if True, break input vertex cells into individual vertex cells
            (one point per cell). If False, the input vertex cells will be ignored.
        :param bool lines: if True, break input polylines into line segments.
            If False, input lines will be ignored and the output will have no lines.
        """
        if self._polydata.GetNumberOfPolys() or self._polydata.GetNumberOfStrips():
            tf = vtk.vtkTriangleFilter()
            tf.SetPassLines(lines)
            tf.SetPassVerts(verts)
            tf.SetInputData(self._polydata)
            tf.Update()
            return self._update(tf.GetOutput())

        elif self._polydata.GetNumberOfLines():
            vct = vtk.vtkContourTriangulator()
            vct.SetInputData(self._polydata)
            vct.Update()
            return self._update(vct.GetOutput())

        else:
            #colors.printc("Error in triangulate()")
            return self



    def addCurvatureScalars(self, method=0, lut=None):
        """
        Add scalars to ``Mesh`` that contains the
        curvature calculated in three different ways.

        :param int method: 0-gaussian, 1-mean, 2-max, 3-min curvature.
        :param lut: optional vtkLookUpTable up table.

        :Example:
            .. code-block:: python

                from vedo import Torus
                Torus().addCurvatureScalars().show()

            |curvature|
        """
        curve = vtk.vtkCurvatures()
        curve.SetInputData(self._polydata)
        curve.SetCurvatureType(method)
        curve.Update()
        self._polydata = curve.GetOutput()

        self._mapper.SetInputData(self._polydata)
        if lut:
            self._mapper.SetLookupTable(lut)
            self._mapper.SetUseLookupTableScalarRange(1)
        self._mapper.Update()
        self.Modified()
        self._mapper.ScalarVisibilityOn()
        return self


    def addElevationScalars(self, lowPoint=(), highPoint=(), vrange=(), lut=None):
        """
        Add to ``Mesh`` a scalar array that contains distance along a specified direction.

        :param list low: one end of the line (small scalar values). Default (0,0,0).
        :param list high: other end of the line (large scalar values). Default (0,0,1).
        :param list vrange: set the range of the scalar. Default is (0, 1).
        :param lut: optional vtkLookUpTable up table (see makeLUT method).

        :Example:
            .. code-block:: python

                from vedo import Sphere

                s = Sphere().addElevationScalars(lowPoint=(0,0,0), highPoint=(1,1,1))
                s.addScalarBar().show(axes=1)

                |elevation|
        """
        ef = vtk.vtkElevationFilter()
        ef.SetInputData(self.polydata())
        if len(lowPoint) == 3:
            ef.SetLowPoint(lowPoint)
        if len(highPoint) == 3:
            ef.SetHighPoint(highPoint)
        if len(vrange) == 2:
            ef.SetScalarRange(vrange)

        ef.Update()
        self._polydata = ef.GetOutput()

        self._mapper.SetInputData(self._polydata)
        if lut:
            self._mapper.SetLookupTable(lut)
            self._mapper.SetUseLookupTableScalarRange(1)
        self._mapper.Update()
        self.Modified()
        self._mapper.ScalarVisibilityOn()
        return self

    def addShadow(self, x=None, y=None, z=None, c=(0.5, 0.5, 0.5), alpha=1, culling=1):
        """
        Generate a shadow out of an ``Mesh`` on one of the three Cartesian planes.
        The output is a new ``Mesh`` representing the shadow.
        This new mesh is accessible through `mesh.shadow`.
        By default the shadow mesh is placed on the bottom/back wall of the bounding box.

        :param float x,y,z: identify the plane to cast the shadow to ['x', 'y' or 'z'].
            The shadow will lay on the orthogonal plane to the specified axis at the
            specified value of either x, y or z.

        :param int culling: choose between front [1] or backface [-1] culling or None.

        |shadow|  |shadow.py|_

        |airplanes| |airplanes.py|_
        """
        if x is not None:
            self.shadowX = x
            shad = self.clone().projectOnPlane('x').x(x)
        elif y is not None:
            self.shadowY = y
            shad = self.clone().projectOnPlane('y').y(y)
        elif z is not None:
            self.shadowZ = z
            shad = self.clone().projectOnPlane('z').z(z)
        else:
            print('Error in addShadow(): must set x, y or z to a float!')
            return self
        shad.c(c).alpha(alpha).wireframe(False).flat()
        if culling==1:
            shad.frontFaceCulling()
        elif culling==-1:
            shad.backFaceCulling()
        shad.GetProperty().LightingOff()
        #shad.SetPickable(False)
        shad.SetUseBounds(True)
        self.shadow = shad
        return self

    def _updateShadow(self):
        p = self.GetPosition()
        if self.shadowX is not None:
            self.shadow.SetPosition(self.shadowX, p[1], p[2])
        elif self.shadowY is not None:
            self.shadow.SetPosition(p[0], self.shadowY, p[2])
        elif self.shadowZ is not None:
            self.shadow.SetPosition(p[0], p[1], self.shadowZ)
        return self


    def subdivide(self, N=1, method=0):
        """Increase the number of vertices of a surface mesh.

        :param int N: number of subdivisions.
        :param int method: Loop(0), Linear(1), Adaptive(2), Butterfly(3)
        """
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputData(self._polydata)
        triangles.Update()
        originalMesh = triangles.GetOutput()
        if method == 0:
            sdf = vtk.vtkLoopSubdivisionFilter()
        elif method == 1:
            sdf = vtk.vtkLinearSubdivisionFilter()
        elif method == 2:
            sdf = vtk.vtkAdaptiveSubdivisionFilter()
        elif method == 3:
            sdf = vtk.vtkButterflySubdivisionFilter()
        else:
            colors.printc("Error in subdivide: unknown method.", c="r")
            raise RuntimeError()
        if method != 2:
            sdf.SetNumberOfSubdivisions(N)
        sdf.SetInputData(originalMesh)
        sdf.Update()
        return self._update(sdf.GetOutput())

    def decimate(self, fraction=0.5, N=None, method='quadric', boundaries=False):
        """
        Downsample the number of vertices in a mesh to `fraction`.

        :param float fraction: the desired target of reduction.
        :param int N: the desired number of final points
            (**fraction** is recalculated based on it).
        :param str method: can be either 'quadric' or 'pro'. In the first case triagulation
            will look like more regular, irrespective of the mesh origianl curvature.
            In the second case triangles are more irregular but mesh is more precise on more
            curved regions.
        :param bool boundaries: (True), in `pro` mode decide whether
            to leave boundaries untouched or not.

        .. note:: Setting ``fraction=0.1`` leaves 10% of the original nr of vertices.

        |skeletonize| |skeletonize.py|_
        """
        poly = self._polydata
        if N:  # N = desired number of points
            Np = poly.GetNumberOfPoints()
            fraction = float(N) / Np
            if fraction >= 1:
                return self

        if 'quad' in method:
            decimate = vtk.vtkQuadricDecimation()
            decimate.SetAttributeErrorMetric(True)
            if self.GetTexture():
                decimate.TCoordsAttributeOn()
            else:
                decimate.SetVolumePreservation(True)
        else:
            decimate = vtk.vtkDecimatePro()
            decimate.PreserveTopologyOn()
            if boundaries:
                decimate.BoundaryVertexDeletionOff()
            else:
                decimate.BoundaryVertexDeletionOn()
        decimate.SetInputData(poly)
        decimate.SetTargetReduction(1 - fraction)
        decimate.Update()
        return self._update(decimate.GetOutput())

    def smoothLaplacian(self, niter=15, relaxfact=0.1, edgeAngle=15, featureAngle=60):
        """
        Adjust mesh point positions using `Laplacian` smoothing.

        :param int niter: number of iterations.
        :param float relaxfact: relaxation factor.
            Small `relaxfact` and large `niter` are more stable.
        :param float edgeAngle: edge angle to control smoothing along edges
            (either interior or boundary).
        :param float featureAngle: specifies the feature angle for sharp edge identification.

        .. hint:: |mesh_smoother1.py|_
        """
        poly = self._polydata
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(cl.GetOutput())
        smoothFilter.SetNumberOfIterations(niter)
        smoothFilter.SetRelaxationFactor(relaxfact)
        smoothFilter.SetEdgeAngle(edgeAngle)
        smoothFilter.SetFeatureAngle(featureAngle)
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.FeatureEdgeSmoothingOn()
        smoothFilter.GenerateErrorScalarsOn()
        smoothFilter.Update()
        return self._update(smoothFilter.GetOutput())

    def smoothWSinc(self, niter=15, passBand=0.1, edgeAngle=15, featureAngle=60):
        """
        Adjust mesh point positions using the `Windowed Sinc` function interpolation kernel.

        :param int niter: number of iterations.
        :param float passBand: set the passband value for the windowed sinc filter.
        :param float edgeAngle: edge angle to control smoothing along edges
             (either interior or boundary).
        :param float featureAngle: specifies the feature angle for sharp edge identification.

        |mesh_smoother1| |mesh_smoother1.py|_
        """
        poly = self._polydata
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        smoothFilter = vtk.vtkWindowedSincPolyDataFilter()
        smoothFilter.SetInputData(cl.GetOutput())
        smoothFilter.SetNumberOfIterations(niter)
        smoothFilter.SetEdgeAngle(edgeAngle)
        smoothFilter.SetFeatureAngle(featureAngle)
        smoothFilter.SetPassBand(passBand)
        smoothFilter.NormalizeCoordinatesOn()
        smoothFilter.NonManifoldSmoothingOn()
        smoothFilter.FeatureEdgeSmoothingOn()
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.Update()
        return self._update(smoothFilter.GetOutput())


    def fillHoles(self, size=None):
        """Identifies and fills holes in input mesh.
        Holes are identified by locating boundary edges, linking them together into loops,
        and then triangulating the resulting loops.

        :param float size: approximate limit to the size of the hole that can be filled.

        Example: |fillholes.py|_
        """
        fh = vtk.vtkFillHolesFilter()
        if not size:
            mb = self.maxBoundSize()
            size = mb / 10
        fh.SetHoleSize(size)
        fh.SetInputData(self._polydata)
        fh.Update()
        return self._update(fh.GetOutput())


    def isInside(self, point, tol=0.0001):
        """
        Return True if point is inside a polydata closed surface.
        """
        poly = self.polydata()
        points = vtk.vtkPoints()
        points.InsertNextPoint(point)
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(points)
        sep = vtk.vtkSelectEnclosedPoints()
        sep.SetTolerance(tol)
        sep.CheckSurfaceOff()
        sep.SetInputData(pointsPolydata)
        sep.SetSurfaceData(poly)
        sep.Update()
        return sep.IsInside(0)


    def insidePoints(self, pts, invert=False, tol=1e-05, returnIds=False):
        """
        Return the point cloud that is inside mesh surface.

        |pca| |pca.py|_
        """
        if isinstance(pts, Points):
            pointsPolydata = pts.polydata()
            pts = pts.points()
        else:
            vpoints = vtk.vtkPoints()
            pts = np.ascontiguousarray(pts)
            vpoints.SetData(numpy_to_vtk(pts, deep=True))
            pointsPolydata = vtk.vtkPolyData()
            pointsPolydata.SetPoints(vpoints)

        sep = vtk.vtkSelectEnclosedPoints()
        sep.SetTolerance(tol)
        sep.SetInputData(pointsPolydata)
        sep.SetSurfaceData(self.polydata())
        sep.SetInsideOut(invert)
        sep.Update()

        mask = Mesh(sep.GetOutput()).getPointArray(0).astype(np.bool)
        ids = np.array(range(len(pts)))[mask]

        if returnIds:
            return ids
        else:
            pcl = Points(pts[ids])
            pcl.name = "insidePoints"
            return pcl

    def boundaries(self,
                   boundaryEdges=True,
                   featureAngle=65,
                   nonManifoldEdges=True,
                   returnPointIds=False,
                   returnCellIds=False,
                   ):
        """
        Return a ``Mesh`` that shows the boundary lines of an input mesh.

        :param bool boundaryEdges: Turn on/off the extraction of boundary edges.
        :param float featureAngle: Specify the feature angle for extracting feature edges.
        :param bool nonManifoldEdges: Turn on/off the extraction of non-manifold edges.
        :param bool returnPointIds: return a numpy array of point indices
        :param bool returnCellIds: return a numpy array of cell indices
        """
        fe = vtk.vtkFeatureEdges()
        fe.SetBoundaryEdges(boundaryEdges)
        fe.SetFeatureAngle(featureAngle)
        fe.SetNonManifoldEdges(nonManifoldEdges)
        fe.ColoringOff()

        if returnPointIds or returnCellIds:

            idf = vtk.vtkIdFilter()
            idf.SetInputData(self.polydata())
            idf.SetIdsArrayName("BoundaryIds")
            idf.SetPointIds(returnPointIds)
            idf.SetCellIds(returnCellIds)
            idf.Update()
            fe.SetInputData(idf.GetOutput())
            fe.ManifoldEdgesOff()
            fe.NonManifoldEdgesOff()
            fe.BoundaryEdgesOn()
            fe.FeatureEdgesOff()
            fe.Update()
            if returnPointIds:
                vid = fe.GetOutput().GetPointData().GetArray("BoundaryIds")
            if returnCellIds:
                vid = fe.GetOutput().GetCellData().GetArray("BoundaryIds")
            npid = vtk_to_numpy(vid).astype(int)
            return npid

        else:

            fe.SetInputData(self.polydata())
            fe.Update()
            return Mesh(fe.GetOutput(), c="p").lw(5).lighting('off')


    def connectedVertices(self, index, returnIds=False):
        """Find all vertices connected to an input vertex specified by its index.

        :param bool returnIds: return vertex IDs instead of vertex coordinates.

        |connVtx| |connVtx.py|_
        """
        poly = self._polydata

        cellIdList = vtk.vtkIdList()
        poly.GetPointCells(index, cellIdList)

        idxs = []
        for i in range(cellIdList.GetNumberOfIds()):
            pointIdList = vtk.vtkIdList()
            poly.GetCellPoints(cellIdList.GetId(i), pointIdList)
            for j in range(pointIdList.GetNumberOfIds()):
                idj = pointIdList.GetId(j)
                if idj == index:
                    continue
                if idj in idxs:
                    continue
                idxs.append(idj)

        if returnIds:
            return idxs
        else:
            trgp = []
            for i in idxs:
                p = [0, 0, 0]
                poly.GetPoints().GetPoint(i, p)
                trgp.append(p)
            return np.array(trgp)


    def connectedCells(self, index, returnIds=False):
        """Find all cellls connected to an input vertex specified by its index."""

        # Find all cells connected to point index
        dpoly = self._polydata
        cellPointIds = vtk.vtkIdList()
        dpoly.GetPointCells(index, cellPointIds)

        ids = vtk.vtkIdTypeArray()
        ids.SetNumberOfComponents(1)
        rids = []
        for k in range(cellPointIds.GetNumberOfIds()):
            cid = cellPointIds.GetId(k)
            ids.InsertNextValue(cid)
            rids.append(int(cid))
        if returnIds:
            return rids

        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(ids)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, dpoly)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(extractSelection.GetOutput())
        gf.Update()
        return Mesh(gf.GetOutput()).lw(1)


    def intersectWithLine(self, p0, p1):
        """Return the list of points intersecting the mesh
        along the segment defined by two points `p0` and `p1`.

        :Example:
            .. code-block:: python

                from vedo import *
                s = Spring(alpha=0.2)
                pts = s.intersectWithLine([0,0,0], [1,0.1,0])
                ln = Line([0,0,0], [1,0.1,0], c='blue')
                ps = Points(pts, r=10, c='r')
                show(s, ln, ps, bg='white')

            |intline|
        """
        if not self.line_locator:
            self.line_locator = vtk.vtkOBBTree()
            self.line_locator.SetDataSet(self.polydata())
            self.line_locator.BuildLocator()

#        t = vtk.mutable(tol)
#        x = [0,0,0]
#        pcoords = [0,0,0]
#        subId = vtk.mutable(0)
#        found = self.line_locator.IntersectWithLine(p0, p1, tol, t, x, pcoords, subId)
#        if found:
#            return [x]
#        else:
#            return []

        intersectPoints = vtk.vtkPoints()
        self.line_locator.IntersectWithLine(p0, p1, intersectPoints, None)
        pts = []
        for i in range(intersectPoints.GetNumberOfPoints()):
            intersection = [0, 0, 0]
            intersectPoints.GetPoint(i, intersection)
            pts.append(intersection)
        return pts


    def silhouette(self, direction=None, borderEdges=True, featureAngle=False):
        """
        Return a new line ``Mesh`` which corresponds to the outer `silhouette`
        of the input as seen along a specified `direction`, this can also be
        a ``vtkCamera`` object.

        :param list direction: viewpoint direction vector.
            If *None* this is guessed by looking at the minimum
            of the sides of the bounding box.
        :param bool borderEdges: enable or disable generation of border edges
        :param float featureAngle: minimal angle for sharp edges detection.
            If set to `False` the functionality is disabled.

        |silhouette| |silhouette.py|_
        """
        sil = vtk.vtkPolyDataSilhouette()
        sil.SetInputData(self.polydata())
        sil.SetBorderEdges(borderEdges)
        if featureAngle == False:
            sil.SetEnableFeatureAngle(0)
        else:
            sil.SetEnableFeatureAngle(1)
            sil.SetFeatureAngle(featureAngle)

        if (direction is None
            and settings.plotter_instance and settings.plotter_instance.camera):
            sil.SetCamera(settings.plotter_instance.camera)
            m = Mesh()
            m._mapper.SetInputConnection(sil.GetOutputPort())

        elif isinstance(direction, vtk.vtkCamera):
            sil.SetCamera(direction)
            m = Mesh()
            m._mapper.SetInputConnection(sil.GetOutputPort())

        elif direction == '2d':
            sil.SetVector(3.4,4.5,5.6) # random
            sil.SetDirectionToSpecifiedVector()
            sil.Update()
            m = Mesh(sil.GetOutput())

        elif utils.isSequence(direction):
            sil.SetVector(direction)
            sil.SetDirectionToSpecifiedVector()
            sil.Update()
            m = Mesh(sil.GetOutput())
        else:
            colors.printc('Error in silhouette(): direction is', [direction], c='r')
            colors.printc(' render the scene with show() or specify camera/direction', c='r')
            return self

        m.lw(2).c((0,0,0)).lighting('off')
        m._mapper.SetResolveCoincidentTopologyToPolygonOffset()
        return m


    def followCamera(self, cam=None):
        """
        Mesh object will follow camera movements and stay locked to it.
        Use ``mesh.followCamera(False)`` to disable it.

        :param vtkCamera cam: if `None` the text will auto-orient itself to the active camera.
            A ``vtkCamera`` object can also be passed.
        """
        if cam is False:
            self.SetCamera(None)
            return self
        if isinstance(cam, vtk.vtkCamera):
            self.SetCamera(cam)
        else:
            # if settings.plotter_instance and settings.plotter_instance.camera:
            #     self.SetCamera(settings.plotter_instance.camera)
            # else:
            self._set2actcam=True
            # colors.printc("Error in followCamera(): needs an already rendered scene,", c='r')
            # colors.printc("                         or passing a vtkCamera object.", c='r')
        return self


    def isobands(self, n=10, vmin=None, vmax=None):
        """
        Return a new ``Mesh`` representing the isobands of the active scalars.
        This is a new mesh where the scalar is now associated to cell faces and
        used to colorize the mesh.

        :param int n: number of isolines in the range
        :param float vmin: minimum of the range
        :param float vmax: maximum of the range

        |isolines| |isolines.py|_
        """
        r0, r1 = self._polydata.GetScalarRange()
        if vmin is None:
            vmin = r0
        if vmax is None:
            vmax = r1

        # --------------------------------
        bands = []
        dx = (vmax - vmin)/float(n)
        b = [vmin, vmin + dx / 2.0, vmin + dx]
        i = 0
        while i < n:
            bands.append(b)
            b = [b[0] + dx, b[1] + dx, b[2] + dx]
            i += 1

        # annotate, use the midpoint of the band as the label
        lut = self.mapper().GetLookupTable()
        labels = []
        for b in bands:
            labels.append('{:4.2f}'.format(b[1]))
        values = vtk.vtkVariantArray()
        for la in labels:
            values.InsertNextValue(vtk.vtkVariant(la))
        for i in range(values.GetNumberOfTuples()):
            lut.SetAnnotation(i, values.GetValue(i).ToString())

        bcf = vtk.vtkBandedPolyDataContourFilter()
        bcf.SetInputData(self.polydata())
        # Use either the minimum or maximum value for each band.
        for i in range(len(bands)):
            bcf.SetValue(i, bands[i][2])
        # We will use an indexed lookup table.
        bcf.SetScalarModeToIndex()
        bcf.GenerateContourEdgesOff()
        bcf.Update()
        bcf.GetOutput().GetCellData().GetScalars().SetName("IsoBands")
        m1 = Mesh(bcf.GetOutput()).computeNormals(cells=True)
        m1.mapper().SetLookupTable(lut)
        return m1


    def isolines(self, n=10, vmin=None, vmax=None):
        """
        Return a new ``Mesh`` representing the isolines of the active scalars.

        :param int n: number of isolines in the range
        :param float vmin: minimum of the range
        :param float vmax: maximum of the range

        |isolines| |isolines.py|_
        """
        bcf = vtk.vtkContourFilter()
        bcf.SetInputData(self.polydata())
        r0, r1 = self._polydata.GetScalarRange()
        if vmin is None:
            vmin = r0
        if vmax is None:
            vmax = r1
        bcf.GenerateValues(n, vmin, vmax)
        bcf.Update()
        sf = vtk.vtkStripper()
        sf.SetJoinContiguousSegments(True)
        sf.SetInputData(bcf.GetOutput())
        sf.Update()
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(sf.GetOutput())
        cl.Update()
        msh = Mesh(cl.GetOutput(), c="k").lighting('off')
        msh._mapper.SetResolveCoincidentTopologyToPolygonOffset()
        return msh


    def extrude(self, zshift=1, rotation=0, dR=0, cap=True, res=1):
        """
        Sweep a polygonal data creating a "skirt" from free edges and lines, and lines from vertices.
        The input dataset is swept around the z-axis to create new polygonal primitives.
        For example, sweeping a line results in a cylindrical shell, and sweeping a circle creates a torus.

        You can control whether the sweep of a 2D object (i.e., polygon or triangle strip)
        is capped with the generating geometry.
        Also, you can control the angle of rotation, and whether translation along the z-axis
        is performed along with the rotation. (Translation is useful for creating "springs").
        You also can adjust the radius of the generating geometry using the "dR" keyword.

        The skirt is generated by locating certain topological features.
        Free edges (edges of polygons or triangle strips only used by one polygon or triangle strips)
        generate surfaces. This is true also of lines or polylines. Vertices generate lines.

        This filter can be used to model axisymmetric objects like cylinders, bottles, and wine glasses;
        or translational/rotational symmetric objects like springs or corkscrews.

        Warning:

            Some polygonal objects have no free edges (e.g., sphere). When swept, this will result
            in two separate surfaces if capping is on, or no surface if capping is off.

        |extrude| |extrude.py|_
        """
        if utils.isSequence(zshift):
#            ms = [] # todo
#            poly0 = self.clone().polydata()
#            for i in range(len(zshift)-1):
#                rf = vtk.vtkRotationalExtrusionFilter()
#                rf.SetInputData(poly0)
#                rf.SetResolution(res)
#                rf.SetCapping(0)
#                rf.SetAngle(rotation)
#                rf.SetTranslation(zshift)
#                rf.SetDeltaRadius(dR)
#                rf.Update()
#                poly1 = rf.GetOutput()
            return self
        else:
            rf = vtk.vtkRotationalExtrusionFilter()
            rf.SetInputData(self.polydata())
            rf.SetResolution(res)
            rf.SetCapping(cap)
            rf.SetAngle(rotation)
            rf.SetTranslation(zshift)
            rf.SetDeltaRadius(dR)
            rf.Update()
            m = Mesh(rf.GetOutput(), c=self.c(), alpha=self.alpha())
            prop = vtk.vtkProperty()
            prop.DeepCopy(self.GetProperty())
            m.SetProperty(prop)
            # assign the same transformation
            m.SetOrigin(self.GetOrigin())
            m.SetScale(self.GetScale())
            m.SetOrientation(self.GetOrientation())
            m.SetPosition(self.GetPosition())
            return m.computeNormals(cells=False).phong()


    def splitByConnectivity(self, maxdepth=1000):
        """
        Split a mesh by connectivity and order the pieces by increasing area.

        :param int maxdepth: only consider this number of mesh parts.

        |splitmesh| |splitmesh.py|_
        """
        self.addIDs()
        pd = self.polydata()
        cf = vtk.vtkConnectivityFilter()
        cf.SetInputData(pd)
        cf.SetExtractionModeToAllRegions()
        cf.ColorRegionsOn()
        cf.Update()
        cpd = cf.GetOutput()
        a = Mesh(cpd)
        alist = []

        for t in range(max(a.getPointArray("RegionId")) - 1):
            if t == maxdepth:
                break
            suba = a.clone().threshold("RegionId", t - 0.1, t + 0.1)
            area = suba.area()
            alist.append([suba, area])

        alist.sort(key=lambda x: x[1])
        alist.reverse()
        blist = []
        for i, l in enumerate(alist):
            l[0].color(i + 1).phong()
            l[0].mapper().ScalarVisibilityOff()
            blist.append(l[0])
        return blist


    def extractLargestRegion(self):
        """
        Extract the largest connected part of a mesh and discard all the smaller pieces.

        .. hint:: |largestregion.py|_
        """
        conn = vtk.vtkConnectivityFilter()
        conn.SetExtractionModeToLargestRegion()
        conn.ScalarConnectivityOff()
        conn.SetInputData(self._polydata)
        conn.Update()
        m = Mesh(conn.GetOutput())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.GetProperty())
        m.SetProperty(pr)
        # assign the same transformation
        m.SetOrigin(self.GetOrigin())
        m.SetScale(self.GetScale())
        m.SetOrientation(self.GetOrientation())
        m.SetPosition(self.GetPosition())
        return m

    def boolean(self, operation, mesh2):
        """Volumetric union, intersection and subtraction of surfaces.

        :param str operation: allowed operations: ``'plus'``, ``'intersect'``, ``'minus'``.

        |boolean| |boolean.py|_
        """
        bf = vtk.vtkBooleanOperationPolyDataFilter()
        poly1 = self.computeNormals().polydata()
        poly2 = mesh2.computeNormals().polydata()
        if operation.lower() == "plus" or operation.lower() == "+":
            bf.SetOperationToUnion()
        elif operation.lower() == "intersect":
            bf.SetOperationToIntersection()
        elif operation.lower() == "minus" or operation.lower() == "-":
            bf.SetOperationToDifference()
        #bf.ReorientDifferenceCellsOn()
        bf.SetInputData(0, poly1)
        bf.SetInputData(1, poly2)
        bf.Update()
        mesh = Mesh(bf.GetOutput(), c=None)
        mesh.name = self.name+operation+mesh2.name
        return mesh


    def intersectWith(self, mesh2, tol=1e-06):
        """
        Intersect this Mesh with the input surface to return a line.

        .. hint:: |surfIntersect.py|_
        """
        bf = vtk.vtkIntersectionPolyDataFilter()
        if isinstance(self, Mesh):
            poly1 = self.polydata()
        else:
            poly1 = self.GetMapper().GetInput()
        if isinstance(mesh2, Mesh):
            poly2 = mesh2.polydata()
        else:
            poly2 = mesh2.GetMapper().GetInput()
        bf.SetInputData(0, poly1)
        bf.SetInputData(1, poly2)
        bf.Update()
        mesh = Mesh(bf.GetOutput(), "k", 1).lighting('off')
        mesh.GetProperty().SetLineWidth(3)
        mesh.name = "surfaceIntersection"
        return mesh


    def implicitModeller(self, distance=0.05, res=(50,50,50),
                         bounds=(), maxdist=None, outer=True):
        """Find the surface which sits at the specified distance from the input one."""
        if not len(bounds):
            bounds = self.bounds()

        if not maxdist:
            maxdist = self.diagonalSize()/2

        imp = vtk.vtkImplicitModeller()
        imp.SetInputData(self.polydata())
        imp.SetSampleDimensions(res)
        imp.SetMaximumDistance(maxdist)
        imp.SetModelBounds(bounds)
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(imp.GetOutputPort())
        contour.SetValue(0, distance)
        contour.Update()
        poly = contour.GetOutput()
        if outer:
            return Mesh(poly).extractLargestRegion().c('lb')
        return Mesh(poly, c='lb')


    def geodesic(self, start, end):
        """Dijkstra algorithm to compute the geodesic line.
        Takes as input a polygonal mesh and performs a single source
        shortest path calculation.

        :param start: start vertex index or close point `[x,y,z]`
        :type start: int, list
        :param end: end vertex index or close point `[x,y,z]`
        :type start: int, list

        |geodesic| |geodesic.py|_
        """
        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()

        if utils.isSequence(start):
            cc = self.points()
            pa = Points(cc)
            start = pa.closestPoint(start, returnIds=True)
            end = pa.closestPoint(end, returnIds=True)
            dijkstra.SetInputData(pa.polydata())
        else:
            dijkstra.SetInputData(self.polydata())

        dijkstra.SetStartVertex(start)
        dijkstra.SetEndVertex(end)
        dijkstra.Update()

        weights = vtk.vtkDoubleArray()
        dijkstra.GetCumulativeWeights(weights)

        length = weights.GetMaxId() + 1
        arr = np.zeros(length)
        for i in range(length):
            arr[i] = weights.GetTuple(i)[0]

        dmesh = Mesh(dijkstra.GetOutput())
        prop = vtk.vtkProperty()
        prop.DeepCopy(self.GetProperty())
        prop.SetLineWidth(3)
        prop.SetOpacity(1)
        dmesh.SetProperty(prop)
        dmesh.info["CumulativeWeights"] = arr
        dmesh.name = "geodesicLine"
        return dmesh















