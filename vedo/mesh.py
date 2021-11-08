import numpy as np
import os
import vtk
import vedo
from vedo.colors import printc, getColor, colorMap
from vedo.utils import isSequence, flatten, mag, buildPolyData, numpy2vtk, vtk2numpy
from vedo.pointcloud import Points
from deprecated import deprecated

__doc__ = ("""Submodule to manage polygonal meshes.""" + vedo.docs._defs)

__all__ = ["Mesh", "merge"]


####################################################
def merge(*meshs, flag=False):
    """
    Build a new mesh formed by the fusion of the input polygonal Meshes (or Points).

    Similar to Assembly, but in this case the input objects become a single mesh entity.

    To keep track of the original identities of the input mesh you can set flag.
    In this case a point array of IDs is added to the merged output mesh.

    .. hint:: |warp1.py|_ |value-iteration.py|_

        |warp1| |value-iteration|
    """
    acts = [a for a in flatten(meshs) if a]

    if not acts:
        return None

    idarr = []
    polyapp = vtk.vtkAppendPolyData()
    for i, a in enumerate(acts):
        try:
            poly = a.polydata()
        except:
            # so a vtkPolydata can also be passed
            poly = a
        polyapp.AddInputData(poly)
        if flag:
            idarr += [i]*poly.GetNumberOfPoints()
    polyapp.Update()
    mpoly = polyapp.GetOutput()

    if flag:
        varr = numpy2vtk(idarr, dtype=np.uint16, name="OriginalMeshID")
        mpoly.GetPointData().AddArray(varr)

    msh = Mesh(mpoly)
    if isinstance(acts[0], vtk.vtkActor):
        cprp = vtk.vtkProperty()
        cprp.DeepCopy(acts[0].GetProperty())
        msh.SetProperty(cprp)
        msh.property = cprp
    return msh


####################################################
class Mesh(Points):
    """
    Build an instance of object ``Mesh`` derived from ``PointCloud``.

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

        self.line_locator = None
        self._current_texture_name = ''  # used by plotter._keypress

        self._mapper.SetInterpolateScalarsBeforeMapping(vedo.settings.interpolateScalarsBeforeMapping)

        if vedo.settings.usePolygonOffset:
            self._mapper.SetResolveCoincidentTopologyToPolygonOffset()
            pof, pou = vedo.settings.polygonOffsetFactor, vedo.settings.polygonOffsetUnits
            self._mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(pof, pou)

        inputtype = str(type(inputobj))

        if inputobj is None:
            pass

        elif isinstance(inputobj, Mesh) or isinstance(inputobj, vtk.vtkActor):
            polyCopy = vtk.vtkPolyData()
            polyCopy.DeepCopy(inputobj.GetMapper().GetInput())
            self._data = polyCopy
            self._mapper.SetInputData(polyCopy)
            self._mapper.SetScalarVisibility(inputobj.GetMapper().GetScalarVisibility())
            pr = vtk.vtkProperty()
            pr.DeepCopy(inputobj.GetProperty())
            self.SetProperty(pr)
            self.property = pr

        elif isinstance(inputobj, vtk.vtkPolyData):
            if inputobj.GetNumberOfCells() == 0:
                carr = vtk.vtkCellArray()
                for i in range(inputobj.GetNumberOfPoints()):
                    carr.InsertNextCell(1)
                    carr.InsertCellPoint(i)
                inputobj.SetVerts(carr)
            self._data = inputobj  # cache vtkPolyData and mapper for speed

        elif isinstance(inputobj, (vtk.vtkStructuredGrid, vtk.vtkRectilinearGrid)):
            if vedo.settings.visibleGridEdges:
                gf = vtk.vtkExtractEdges()
                gf.SetInputData(inputobj)
            else:
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(inputobj)
            gf.Update()
            self._data = gf.GetOutput()

        elif "trimesh" in inputtype:
            tact = vedo.utils.trimesh2vedo(inputobj, alphaPerCell=False)
            self._data = tact.polydata()

        elif "meshio" in inputtype: # meshio-4.0.11
            if len(inputobj.cells):
                mcells = []
                for cellblock in inputobj.cells:
                    if cellblock.type in ("triangle", "quad"):
                        mcells += cellblock.data.tolist()
                self._data = buildPolyData(inputobj.points, mcells)
            else:
                self._data = buildPolyData(inputobj.points, None)
            # add arrays:
            try:
                if len(inputobj.point_data):
                    for k in inputobj.point_data.keys():
                        vdata = numpy2vtk(inputobj.point_data[k])
                        vdata.SetName(str(k))
                        self._data.GetPointData().AddArray(vdata)
            except AssertionError:
                print("Could not add meshio point data, skip.")
            try:
                if len(inputobj.cell_data):
                    for k in inputobj.cell_data.keys():
                        vdata = numpy2vtk(inputobj.cell_data[k])
                        vdata.SetName(str(k))
                        self._data.GetCellData().AddArray(vdata)
            except AssertionError:
                print("Could not add meshio cell data, skip.")

        elif "meshlab" in inputtype:
            self._data = vedo.utils._meshlab2vedo(inputobj)

        elif isSequence(inputobj):
            ninp = len(inputobj)
            if ninp == 0:
                self._data = vtk.vtkPolyData()
            elif ninp == 2:  # assume [vertices, faces]
                self._data = buildPolyData(inputobj[0], inputobj[1])
            else:            # assume [vertices] or vertices
                self._data = buildPolyData(inputobj, None)

        elif hasattr(inputobj, "GetOutput"):  # passing vtk object
            if hasattr(inputobj, "Update"): inputobj.Update()
            if isinstance(inputobj.GetOutput(), vtk.vtkPolyData):
                self._data = inputobj.GetOutput()
            else:
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(inputobj.GetOutput())
                gf.Update()
                self._data = gf.GetOutput()

        elif isinstance(inputobj, str):
            dataset = vedo.io.load(inputobj)
            self.filename = inputobj
            if "TetMesh" in str(type(dataset)):
                self._data = dataset.tomesh().polydata()
            else:
                self._data = dataset.polydata()

        else:
            try:
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(inputobj)
                gf.Update()
                self._data = gf.GetOutput()
            except:
                printc("Error: cannot build mesh from type:\n", inputtype, c='r')
                raise RuntimeError()


        if vedo.settings.computeNormals is not None:
            computeNormals = vedo.settings.computeNormals

        if self._data:
            if computeNormals:
                pdnorm = vtk.vtkPolyDataNormals()
                pdnorm.SetInputData(self._data)
                pdnorm.ComputePointNormalsOn()
                pdnorm.ComputeCellNormalsOn()
                pdnorm.FlipNormalsOff()
                pdnorm.ConsistencyOn()
                pdnorm.Update()
                self._data = pdnorm.GetOutput()

        self._mapper.SetInputData(self._data)

        self._bfprop = None  # backface property holder

        self.property = self.GetProperty()
        self.property.SetInterpolationToPhong()

        # set the color by c or by scalar
        if self._data:

            arrexists = False

            if c is None:
                ptdata = self._data.GetPointData()
                cldata = self._data.GetCellData()
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
                    c = getColor(c)
                elif isinstance(c, float) and c<=1:
                    c = colorMap(c, "rainbow", 0,1)
                else:
                    c = getColor(c)
                self.property.SetColor(c)
                self.property.SetAmbient(0.1)
                self.property.SetDiffuse(1)
                self.property.SetSpecular(.05)
                self.property.SetSpecularPower(5)
                self._mapper.ScalarVisibilityOff()

        if alpha is not None:
            self.property.SetOpacity(alpha)
        return


    def faces(self):
        """
        Get cell polygonal connectivity ids as a python ``list``.
        The output format is: [[id0 ... idn], [id0 ... idm],  etc].
        """
        arr1d = vtk2numpy(self._data.GetPolys().GetData())
        if arr1d is None:
            return []

        #Get cell connettivity ids as a 1D array. vtk format is:
        #[nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        if len(arr1d) == 0:
            arr1d = vtk2numpy(self._data.GetStrips().GetData())
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

    def cells(self):
        """Alias for ``faces()``."""
        return self.faces()


    def lines(self, flat=False):
        """Get lines connectivity ids as a numpy array.
        Default format is [[id0,id1], [id3,id4], ...]

        :param bool flat: 1D numpy array as [2, 10,20, 3, 10,11,12, 2, 70,80, ...]
        """
        #Get cell connettivity ids as a 1D array. The vtk format is:
        #    [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        arr1d = vtk2numpy(self.polydata(False).GetLines().GetData())

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

    def texture(self, tname='',
                tcoords=None,
                interpolate=True,
                repeat=True,
                edgeClamp=False,
                scale=None,
                ushift=None,
                vshift=None,
                seamThreshold=None,
                ):
        """
        Assign a texture to mesh from image file or predefined texture `tname`.
        If tname is set to ``None`` texture is disabled.
        If tname is set to '' then a png or jpg file is looked for with same name and path.
        Input tname can also be an array of shape (n,m,3).

        :param bool interpolate: turn on/off linear interpolation of the texture map when rendering.
        :param bool repeat: repeat of the texture when tcoords extend beyond the [0,1] range.
        :param bool edgeClamp: turn on/off the clamping of the texture map when
            the texture coords extend beyond the [0,1] range.
            Only used when repeat is False, and edge clamping is supported by the graphics card.

        :param bool scale: scale the texture image by this factor
        :param bool ushift: shift u-coordinates of texture by this amaount
        :param bool vshift: shift v-coordinates of texture by this amaount
        :param float seamThreshold: try to seal seams in texture by collapsing triangles
            (test values around 1.0, lower values = stronger collapse)
        """
        pd = self.polydata(False)
        if tname is None:
            pd.GetPointData().SetTCoords(None)
            pd.GetPointData().Modified()
            return self
            ###########

        if isinstance(tname, str) and 'https' in tname:
            tname = vedo.io.download(tname, verbose=False)

        if isSequence(tname):
            from PIL import Image
            from tempfile import NamedTemporaryFile
            tmp_file = NamedTemporaryFile()
            im = Image.fromarray(tname)
            im.save(tmp_file.name+".bmp")
            tname = tmp_file.name+".bmp"

        if tname == '':
            ext = os.path.basename(str(self.filename)).split('.')[-1]
            tname = str(self.filename).replace('.'+ext, '.png')
            if not os.path.isfile(tname):
                tname = str(self.filename).replace('.'+ext, '.jpg')
            if not os.path.isfile(tname):
                printc("Error in texture(): default texture file must be png or jpg",
                       "\n e.g.", tname, c='r')
                raise RuntimeError()

        if isinstance(tname, vtk.vtkTexture):
            tu = tname
        else:
            if tcoords is not None:
                if not isinstance(tcoords, np.ndarray):
                    tcoords = np.array(tcoords)
                if tcoords.ndim != 2:
                    printc('tcoords must be a 2-dimensional array', c='r')
                    return self
                if tcoords.shape[0] != pd.GetNumberOfPoints():
                    printc('Error in texture(): nr of texture coords must match nr of points', c='r')
                    return self
                if tcoords.shape[1] != 2:
                    printc('Error in texture(): vector must have 2 components', c='r')
                tarr = numpy2vtk(tcoords)
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
                        ntc = vtk2numpy(tc)
                        if scale:  ntc *= scale
                        if ushift: ntc[:,0] += ushift
                        if vshift: ntc[:,1] += vshift
                        tc = numpy2vtk(tc)
                    pd.GetPointData().SetTCoords(tc)
                    pd.GetPointData().Modified()

            fn = vedo.settings.textures_path + tname + ".jpg"
            if os.path.exists(tname):
                fn = tname
            elif not os.path.exists(fn):
                printc("File does not exist or texture", tname,
                       "not found in", vedo.settings.textures_path, c="r")
                printc("\tin Available built-in textures:", c="m", end=" ")
                for ff in os.listdir(vedo.settings.textures_path):
                    printc(ff.split(".")[0], end=" ", c="m")
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
                printc("Error in texture(): supported files, PNG, BMP or JPG", c="r")
                return self
            reader.SetFileName(fn)
            reader.Update()

            tu = vtk.vtkTexture()
            tu.SetInputData(reader.GetOutput())
            tu.SetInterpolate(interpolate)
            tu.SetRepeat(repeat)
            tu.SetEdgeClamp(edgeClamp)

        self.property.SetColor(1, 1, 1)
        self._mapper.ScalarVisibilityOff()
        self.SetTexture(tu)

        if seamThreshold is not None:
            tname = self._data.GetPointData().GetTCoords().GetName()
            grad = self.gradient(tname)
            ugrad, vgrad = np.split(grad, 2, axis=1)
            ugradm, vgradm = vedo.utils.mag2(ugrad), vedo.utils.mag2(vgrad)
            gradm = np.log(ugradm + vgradm)
            largegrad_ids = np.arange(len(grad))[gradm>seamThreshold*4]
            uvmap = self.pointdata[tname]
            # collapse triangles that have large gradient
            new_points = self.points(transformed=False)
            for f in self.faces():
                if np.isin(f, largegrad_ids).all():
                    id1, id2, id3 = f
                    uv1, uv2, uv3 = uvmap[f]
                    d12 = vedo.mag2(uv1-uv2)
                    d23 = vedo.mag2(uv2-uv3)
                    d31 = vedo.mag2(uv3-uv1)
                    idm = np.argmin([d12, d23, d31])
                    if idm == 0:
                        new_points[id1] = new_points[id3]
                        new_points[id2] = new_points[id3]
                    elif idm == 1:
                        new_points[id2] = new_points[id1]
                        new_points[id3] = new_points[id1]
            self.points(new_points)

        self.Modified()
        return self


    def computeNormals(self, points=True, cells=True, featureAngle=None, consistency=True):
        """Compute cell and vertex normals for the mesh.

        :param bool points: do the computation for the vertices
        :param bool cells: do the computation for the cells

        :param float featureAngle: specify the angle that defines a sharp edge.
            If the difference in angle across neighboring polygons is greater than this value,
            the shared edge is considered "sharp" and it is splitted.

        :param bool consistency: turn on/off the enforcement of consistent polygon ordering.

        .. warning:: if featureAngle is set to a float the Mesh can be modified, and it
            can have a different nr. of vertices from the original.
        """
        poly = self.polydata(False)
        pdnorm = vtk.vtkPolyDataNormals()
        pdnorm.SetInputData(poly)
        pdnorm.SetComputePointNormals(points)
        pdnorm.SetComputeCellNormals(cells)
        pdnorm.SetConsistency(consistency)
        pdnorm.FlipNormalsOff()
        if featureAngle:
            pdnorm.SetSplitting(True)
            pdnorm.SetFeatureAngle(featureAngle)
        else:
            pdnorm.SetSplitting(False)
        # print(pdnorm.GetNonManifoldTraversal())
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
            self.property.SetRepresentationToWireframe()
        else:
            self.property.SetRepresentationToSurface()
        return self

    def flat(self):
        """Set surface interpolation to Flat.

        |wikiphong|
        """
        self.property.SetInterpolationToFlat()
        return self

    def phong(self):
        """Set surface interpolation to Phong."""
        self.property.SetInterpolationToPhong()
        return self

    def backFaceCulling(self, value=True):
        """Set culling of polygons based on orientation
        of normal with respect to camera."""
        self.property.SetBackfaceCulling(value)
        return self

    def renderLinesAsTubes(self, value=True):
        self.property.SetRenderLinesAsTubes(value)
        return self

    def frontFaceCulling(self, value=True):
        """Set culling of polygons based on orientation of normal with respect to camera."""
        self.property.SetFrontfaceCulling(value)
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

        if self.property.GetOpacity() < 1:
            # printc("In backColor(): only active for alpha=1", c="y")
            return self

        if not backProp:
            backProp = vtk.vtkProperty()

        backProp.SetDiffuseColor(getColor(bc))
        backProp.SetOpacity(self.property.GetOpacity())
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
                self.property.EdgeVisibilityOff()
                self.property.SetRepresentationToSurface()
                return self
            self.property.EdgeVisibilityOn()
            self.property.SetLineWidth(lw)
        else:
            return self.property.GetLineWidth()
        return self

    def lw(self, lineWidth=None):
        """Set/get width of mesh edges. Same as `lineWidth()`."""
        return self.lineWidth(lineWidth)

    def lineColor(self, lc=None):
        """Set/get color of mesh edges. Same as `lc()`."""
        if lc is not None:
#            if "ireframe" in self.property.GetRepresentationAsString():
#                self.property.EdgeVisibilityOff()
#                self.color(lc)
#                return self
            self.property.EdgeVisibilityOn()
            self.property.SetEdgeColor(getColor(lc))
        else:
            return self.property.GetEdgeColor()
        return self

    def lc(self, lineColor=None):
        """Set/get color of mesh edges. Same as `lineColor()`."""
        return self.lineColor(lineColor)

    def volume(self):
        """Get/set the volume occupied by mesh."""
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        return mass.GetVolume()

    def area(self):
        """Get/set the surface area of mesh.

        .. hint:: |largestregion.py|_
        """
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        return mass.GetSurfaceArea()

    def isClosed(self):
        """Return ``True`` if mesh is watertight."""
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.BoundaryEdgesOn()
        featureEdges.FeatureEdgesOff()
        featureEdges.NonManifoldEdgesOn()
        featureEdges.SetInputData(self.polydata(False))
        featureEdges.Update()
        ne = featureEdges.GetOutput().GetNumberOfCells()
        return not bool(ne)


    def shrink(self, fraction=0.85):
        """Shrink the triangle polydata in the representation of the input mesh.

        Example:
            .. code-block:: python

                from vedo import *
                pot = load(dataurl+'teapot.vtk').shrink(0.75)
                s = Sphere(r=0.2).pos(0,0,-0.5)
                show(pot, s)

            |shrink| |shrink.py|_
        """
        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputData(self._data)
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
            printc('Error in stretch(): Please define vectors', c='r')
            printc('   mesh.base and mesh.top at creation.', c='r')
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
            if bounds[0] is None: bounds[0] = x0
            if bounds[1] is None: bounds[1] = x1
            if bounds[2] is None: bounds[2] = y0
            if bounds[3] is None: bounds[3] = y1
            if bounds[4] is None: bounds[4] = z0
            if bounds[5] is None: bounds[5] = z1
        cu.SetBounds(bounds)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self._data)
        clipper.SetClipFunction(cu)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())
        return self

    def cutWithPointLoop(self,
                          points,
                          invert=False,
                          on='points',
                          includeBoundary=False,
        ):
        """
        Cut an ``Mesh`` object with a set of points forming a closed loop.

        :param bool invert: invert selection (inside-out)
        :param str on: if 'cells' will extract the whole cells lying inside
            (or outside) the point loop

        :param bool includeBoundary: include cells lying exactly on the
            boundary line. Only relevant on 'cells' mode.
        """
        if isinstance(points, Points):
            vpts = points.polydata().GetPoints()
            points = points.points()
        else:
            vpts = vtk.vtkPoints()
            if len(points[0])==2: # make it 3d
                points = np.asarray(points)
                points = np.c_[points, np.zeros(len(points))]
            for p in points:
                vpts.InsertNextPoint(p)

        if 'cell' in on:
            ippd = vtk.vtkImplicitSelectionLoop()
            ippd.SetLoop(vpts)
            ippd.AutomaticNormalGenerationOn()
            clipper = vtk.vtkExtractPolyDataGeometry()
            clipper.SetInputData(self.polydata())
            clipper.SetImplicitFunction(ippd)
            clipper.SetExtractInside(not invert)
            clipper.SetExtractBoundaryCells(includeBoundary)
        else:
            spol = vtk.vtkSelectPolyData()
            spol.SetLoop(vpts)
            spol.GenerateSelectionScalarsOn()
            spol.GenerateUnselectedOutputOff()
            spol.SetInputData(self.polydata())
            spol.Update()
            clipper = vtk.vtkClipPolyData()
            clipper.SetInputData(spol.GetOutput())
            clipper.SetInsideOut(not invert)
            clipper.SetValue(0.0)
        clipper.Update()
        cpoly = clipper.GetOutput()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _data is
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
        poly = self._data

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

        rev = vtk.vtkReverseSense()
        rev.ReverseCellsOn()
        rev.SetInputData(boundaryPoly)
        rev.Update()

        tf = vtk.vtkTriangleFilter()
        tf.SetInputData(rev.GetOutput())
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
            return self._update(polyapp.GetOutput()).clean()


    def join(self, polys=True, reset=False):
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
        into triangles and lines prior to applying join().

        :param bool polys: polygonal segments will be joined if they are contiguous
        :param bool reset: reset points ordering

        :Warning:

            If triangle strips or polylines exist in the input data
            they will be passed through to the output data.
            This filter will only construct triangle strips if triangle polygons
            are available; and will only construct polylines if lines are available.

        :Example:
            .. code-block:: python

                from vedo import *
                c1 = Cylinder(pos=(0,0,0), r=2, height=3, axis=(1,.0,0), alpha=.1).triangulate()
                c2 = Cylinder(pos=(0,0,2), r=1, height=2, axis=(0,.3,1), alpha=.1).triangulate()
                intersect = c1.intersectWith(c2).join(reset=True)
                spline = Spline(intersect).c('blue').lw(5)
                show(c1, c2, spline, intersect.labels('id'), axes=1)
        """
        sf = vtk.vtkStripper()
        sf.SetPassThroughCellIds(True)
        sf.SetPassThroughPointIds(True)
        sf.SetJoinContiguousSegments(polys)
        sf.SetInputData(self.polydata(False))
        sf.Update()
        if reset:
            poly = sf.GetOutput()
            cpd = vtk.vtkCleanPolyData()
            cpd.PointMergingOn()
            cpd.ConvertLinesToPointsOn()
            cpd.ConvertPolysToLinesOn()
            cpd.ConvertStripsToPolysOn()
            cpd.SetInputData(poly)
            cpd.Update()
            poly = cpd.GetOutput()
            vpts = poly.GetCell(0).GetPoints().GetData()
            poly.GetPoints().SetData(vpts)
            return self._update(poly)
        else:
            return self._update(sf.GetOutput())


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
        if self._data.GetNumberOfPolys() or self._data.GetNumberOfStrips():
            tf = vtk.vtkTriangleFilter()
            tf.SetPassLines(lines)
            tf.SetPassVerts(verts)
            tf.SetInputData(self._data)
            tf.Update()
            return self._update(tf.GetOutput())

        elif self._data.GetNumberOfLines():
            vct = vtk.vtkContourTriangulator()
            vct.SetInputData(self._data)
            vct.Update()
            return self._update(vct.GetOutput())

        else:
            #printc("Error in triangulate()")
            return self

    @deprecated(reason=vedo.colors.red+"Please use distanceTo()"+vedo.colors.reset)
    def distanceToMesh(self, mesh, signed=False, negate=False):
        return self.distanceTo(mesh, signed=signed, negate=negate)

    def distanceTo(self, mesh, signed=False, negate=False):
        '''
        Computes the (signed) distance from one mesh to another.

        |distance2mesh| |distance2mesh.py|_
        '''
        # overrides pointcloud.distanceToMesh()
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

    def addCellArea(self, name="Area"):
        """Add to this mesh a cell data array containing the areas of the polygonal faces"""
        csf = vtk.vtkCellSizeFilter()
        csf.SetInputData(self.polydata(False))
        csf.SetComputeArea(True)
        csf.SetComputeVolume(False)
        csf.SetComputeLength(False)
        csf.SetComputeVertexCount(False)
        csf.SetAreaArrayName(name)
        csf.Update()
        return self._update(csf.GetOutput())


    def addCellVertexCount(self, name="VertexCount"):
        """Add to this mesh a cell data array containing the nr of vertices that a polygonal face has."""
        csf = vtk.vtkCellSizeFilter()
        csf.SetInputData(self.polydata(False))
        csf.SetComputeArea(False)
        csf.SetComputeVolume(False)
        csf.SetComputeLength(False)
        csf.SetComputeVertexCount(True)
        csf.SetVertexCountArrayName(name)
        csf.Update()
        return self._update(csf.GetOutput())


    def addArcLength(self, mesh, name="ArcLength"):
        """Given a mesh, add the length of the arc intersecting each point of the line."""
        arcl = vtk.vtkAppendArcLength()
        arcl.SetInputData(mesh.polydata())
        arcl.Update()
        return self._update(arcl.GetOutput())


    def addQuality(self, measure=6):
        """
        Calculate functions of quality for the elements of a triangular mesh.
        This method adds to the mesh a cell array named "Quality".
        See class `vtkMeshQuality <https://vtk.org/doc/nightly/html/classvtkMeshQuality.html>`_
        for explanation.

        :param int measure: type of estimator

            - EDGE RATIO, 0
            - ASPECT RATIO, 1
            - RADIUS RATIO, 2
            - ASPECT FROBENIUS, 3
            - MED ASPECT FROBENIUS, 4
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
        self._update(pd)
        return self


    def addCurvatureScalars(self, method=0):
        """
        Add scalars to ``Mesh`` that contains the
        curvature calculated in three different ways.

        :param int method: 0-gaussian, 1-mean, 2-max, 3-min curvature.
        :param lut: optional vtkLookUpTable up table.

        :Example:
            .. code-block:: python

                from vedo import Torus
                Torus().addCurvatureScalars().addScalarBar().show(axes=1)

            |curvature|
        """
        curve = vtk.vtkCurvatures()
        curve.SetInputData(self._data)
        curve.SetCurvatureType(method)
        curve.Update()
        self._update(curve.GetOutput())
        self._mapper.ScalarVisibilityOn()
        return self

    def addConnectivity(self):
        """
        Flag a mesh by connectivity: each disconnected region will receive a different Id.
        You can access the array of ids through ``mesh.pointdata["RegionId"]``.
        """
        cf = vtk.vtkConnectivityFilter()
        cf.SetInputData(self.polydata(False))
        cf.SetExtractionModeToAllRegions()
        cf.ColorRegionsOn()
        cf.Update()
        return self._update(cf.GetOutput())


    def addElevationScalars(self, lowPoint=(0,0,0), highPoint=(0,0,1), vrange=(0,1)):
        """
        Add to ``Mesh`` a scalar array that contains distance along a specified direction.

        :param list lowPoint: one end of the line (small scalar values). Default (0,0,0).
        :param list highPoint: other end of the line (large scalar values). Default (0,0,1).
        :param list vrange: set the range of the scalar. Default is (0, 1).

        :Example:
            .. code-block:: python

                from vedo import Sphere
                s = Sphere().addElevationScalars(lowPoint=(0,0,0), highPoint=(1,1,1))
                s.addScalarBar().show(axes=1)

            |elevation|
        """
        ef = vtk.vtkElevationFilter()
        ef.SetInputData(self.polydata())
        ef.SetLowPoint(lowPoint)
        ef.SetHighPoint(highPoint)
        ef.SetScalarRange(vrange)
        ef.Update()
        self._update(ef.GetOutput())
        self._mapper.ScalarVisibilityOn()
        return self


    def addShadow(self, x=None, y=None, z=None, c=(0.6,0.6,0.6), alpha=1, culling=1):
        """
        Generate a shadow out of an ``Mesh`` on one of the three Cartesian planes.
        The output is a new ``Mesh`` representing the shadow.
        This new mesh is accessible through `mesh.shadow`.
        By default the shadow mesh is placed on the bottom wall of the bounding box.

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
        shad.SetPickable(False)
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


    def subdivide(self, N=1, method=0, mel=None):
        """Increase the number of vertices of a surface mesh.

        :param int N: number of subdivisions.
        :param int method: Loop(0), Linear(1), Adaptive(2), Butterfly(3)
        :param float mel: Maximum Edge Length for Adaptive method only.
        """
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputData(self._data)
        triangles.Update()
        originalMesh = triangles.GetOutput()
        if method == 0:
            sdf = vtk.vtkLoopSubdivisionFilter()
        elif method == 1:
            sdf = vtk.vtkLinearSubdivisionFilter()
        elif method == 2:
            sdf = vtk.vtkAdaptiveSubdivisionFilter()
            if mel is None:
                mel = self.diagonalSize() / np.sqrt(self._data.GetNumberOfPoints())/N
            sdf.SetMaximumEdgeLength(mel)
        elif method == 3:
            sdf = vtk.vtkButterflySubdivisionFilter()
        else:
            printc("Error in subdivide: unknown method.", c="r")
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
        poly = self._data
        if N:  # N = desired number of points
            Np = poly.GetNumberOfPoints()
            fraction = float(N) / Np
            if fraction >= 1:
                return self

        if 'quad' in method:
            decimate = vtk.vtkQuadricDecimation()
            # decimate.SetAttributeErrorMetric(True)
            # if self.GetTexture():
            #     decimate.TCoordsAttributeOn()
            # else:
            #     pass
            # decimate.SetVolumePreservation(True)
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

    @deprecated(reason=vedo.colors.red+"Please use smooth()"+vedo.colors.reset)
    def smoothLaplacian(self, niter=15, relaxfact=0.1, edgeAngle=15, featureAngle=60, boundary=False):
        return self.smooth(niter, passBand=0.1, edgeAngle=edgeAngle, boundary=boundary)

    def smooth(self, niter=15, passBand=0.1, edgeAngle=15, featureAngle=60, boundary=False):
        """
        Adjust mesh point positions using the `Windowed Sinc` function interpolation kernel.

        :param int niter: number of iterations.
        :param float passBand: set the passband value for the windowed sinc filter.
        :param float edgeAngle: edge angle to control smoothing along edges
             (either interior or boundary).
        :param float featureAngle: specifies the feature angle for sharp edge identification.

        |mesh_smoother1| |mesh_smoother1.py|_
        """
        poly = self._data
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
        smoothFilter.SetBoundarySmoothing(boundary)
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
        fh.SetInputData(self._data)
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
            vpoints.SetData(numpy2vtk(pts, dtype=float))
            pointsPolydata = vtk.vtkPolyData()
            pointsPolydata.SetPoints(vpoints)

        sep = vtk.vtkSelectEnclosedPoints()
        sep.SetTolerance(tol)
        sep.SetInputData(pointsPolydata)
        sep.SetSurfaceData(self.polydata())
        sep.SetInsideOut(invert)
        sep.Update()

        mask = Mesh(sep.GetOutput()).pointdata[0].astype(np.bool)
        ids = np.array(range(len(pts)))[mask]

        if returnIds:
            return ids
        else:
            pcl = Points(pts[ids])
            pcl.name = "insidePoints"
            return pcl

    def boundaries(self,
                   boundaryEdges=True,
                   nonManifoldEdges=False,
                   featureAngle=180,
                   returnPointIds=False,
                   returnCellIds=False,
        ):
        """
        Return a ``Mesh`` that shows the boundary lines of an input mesh.

        :param bool boundaryEdges: Turn on/off the extraction of boundary edges.
        :param bool nonManifoldEdges: Turn on/off the extraction of non-manifold edges.
        :param float featureAngle: Specify the min angle btw 2 faces for extracting edges.
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
            npid = vtk2numpy(vid).astype(int)
            return npid

        else:

            fe.SetInputData(self.polydata())
            fe.Update()
            return Mesh(fe.GetOutput(), c="p").lw(5).lighting('off')


    def imprint(self, loopline, tol=0.01):
        """
        Imprint the contact surface of one object onto another surface.

        Parameters
        ----------
        loopline : vedo.shapes.Line
            a Line object to be imprinted onto the mesh.
        tol : TYPE, optional
            projection tolerance which controls how close the imprint surface must be to the target.
            The default is 0.01.

        :Example:

            .. code-block:: python

                from vedo import *
                grid = Grid()#.triangulate()
                circle = Circle(r=0.3, res=24).pos(0.11,0.12)
                line = Line(circle, closed=True, lw=4, c='r4')
                grid.imprint(line)
                show(grid, line, axes=1)
        """
        loop = vtk.vtkContourLoopExtraction()
        loop.SetInputData(loopline.polydata())
        loop.Update()

        cleanLoop = vtk.vtkCleanPolyData()
        cleanLoop.SetInputData(loop.GetOutput())
        cleanLoop.Update()

        imp = vtk.vtkImprintFilter()
        imp.SetTargetData(self.polydata())
        imp.SetImprintData(cleanLoop.GetOutput())
        imp.SetTolerance(tol)
        imp.BoundaryEdgeInsertionOn()
        imp.TriangulateOutputOn()
        imp.Update()
        return self._update(imp.GetOutput())


    def connectedVertices(self, index, returnIds=False):
        """Find all vertices connected to an input vertex specified by its index.

        :param bool returnIds: return vertex IDs instead of vertex coordinates.

        |connVtx| |connVtx.py|_
        """
        poly = self._data

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
        dpoly = self._data
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

    def intersectWithLine(self, p0, p1=None, returnIds=False, tol=0):
        """Return the list of points intersecting the mesh
        along the segment defined by two points `p0` and `p1`.

        :param bool returnIds: return the cell ids instead of point coords
        :param float tol: tolerance/precision of the computation (0 = auto).

        :Example:
            .. code-block:: python

                from vedo import *
                s = Spring()
                pts = s.intersectWithLine([0,0,0], [1,0.1,0])
                ln = Line([0,0,0], [1,0.1,0], c='blue')
                ps = Points(pts, r=10, c='r')
                show(s, ln, ps, bg='white')

            |intline|
        """
        if isinstance(p0, Points):
            p0, p1 = p0.points()

        if not self.line_locator:
            self.line_locator = vtk.vtkOBBTree()
            self.line_locator.SetDataSet(self.polydata())
            if not tol:
                tol = mag(np.asarray(p1)-np.asarray(p0))/10000
            self.line_locator.SetTolerance(tol)
            self.line_locator.BuildLocator()

        intersectPoints = vtk.vtkPoints()
        idlist = vtk.vtkIdList()
        self.line_locator.IntersectWithLine(p0, p1, intersectPoints, idlist)
        pts = []
        for i in range(intersectPoints.GetNumberOfPoints()):
            intersection = [0, 0, 0]
            intersectPoints.GetPoint(i, intersection)
            pts.append(intersection)
        pts = np.array(pts)

        if returnIds:
            pts_ids = []
            for i in range(idlist.GetNumberOfIds()):
                cid = idlist.GetId(i)
                pts_ids.append([pts[i], cid])
            return np.array(pts_ids)
        else:
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
        if featureAngle is False:
            sil.SetEnableFeatureAngle(0)
        else:
            sil.SetEnableFeatureAngle(1)
            sil.SetFeatureAngle(featureAngle)

        if (direction is None
            and vedo.settings.plotter_instance
            and vedo.settings.plotter_instance.camera):
            sil.SetCamera(vedo.settings.plotter_instance.camera)
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

        elif isSequence(direction):
            sil.SetVector(direction)
            sil.SetDirectionToSpecifiedVector()
            sil.Update()
            m = Mesh(sil.GetOutput())
        else:
            printc('Error in silhouette(): direction is', [direction], c='r')
            printc(' render the scene with show() or specify camera/direction', c='r')
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
            plt = vedo.settings.plotter_instance
            if plt and plt.camera:
                self.SetCamera(plt.camera)
            else:
                # postpone to show() call
                self._set2actcam=True
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
        r0, r1 = self._data.GetScalarRange()
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
        r0, r1 = self._data.GetScalarRange()
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
        if isSequence(zshift):
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
            # rf = vtk.vtkLinearExtrusionFilter()
            rf.SetInputData(self.polydata(False)) #must not be transformed
            rf.SetResolution(res)
            rf.SetCapping(cap)
            rf.SetAngle(rotation)
            rf.SetTranslation(zshift)
            rf.SetDeltaRadius(dR)
            rf.Update()
            m = Mesh(rf.GetOutput(), c=self.c(), alpha=self.alpha())
            prop = vtk.vtkProperty()
            prop.DeepCopy(self.property)
            m.SetProperty(prop)
            m.property = prop
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

        :param bool addRegions

        |splitmesh| |splitmesh.py|_
        """
        pd = self.polydata(False)
        cf = vtk.vtkConnectivityFilter()
        cf.SetInputData(pd)
        cf.SetExtractionModeToAllRegions()
        cf.ColorRegionsOn()
        cf.Update()
        a = Mesh(cf.GetOutput())
        alist = []

        for t in range(max(a.pointdata["RegionId"]) + 1):
            if t == maxdepth:
                break
            suba = a.clone().threshold("RegionId", t - 0.1, t + 0.1)
            area = suba.area()
            # print('splitByConnectivity  piece:', t, ' area:', area, ' N:',suba.N())
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
        conn.SetInputData(self._data)
        conn.Update()
        m = Mesh(conn.GetOutput())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.property)
        m.SetProperty(pr)
        m.property = pr
        # assign the same transformation
        m.SetOrigin(self.GetOrigin())
        m.SetScale(self.GetScale())
        m.SetOrientation(self.GetOrientation())
        m.SetPosition(self.GetPosition())
        vis = self._mapper.GetScalarVisibility()
        m._mapper.SetScalarVisibility(vis)
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
        mesh.flat()
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


    def geodesic(self, start, end):
        """Dijkstra algorithm to compute the geodesic line.
        Takes as input a polygonal mesh and performs a single source
        shortest path calculation.

        :param int,list start: start vertex index or close point `[x,y,z]`
        :param int,list end: end vertex index or close point `[x,y,z]`

        |geodesic| |geodesic.py|_
        """
        if isSequence(start):
            cc = self.points()
            pa = Points(cc)
            start = pa.closestPoint(start, returnPointId=True)
            end   = pa.closestPoint(end,   returnPointId=True)

        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(self.polydata())
        dijkstra.SetStartVertex(end) # inverted in vtk
        dijkstra.SetEndVertex(start)
        dijkstra.Update()

        weights = vtk.vtkDoubleArray()
        dijkstra.GetCumulativeWeights(weights)

        idlist = dijkstra.GetIdList()
        ids = [idlist.GetId(i) for i in range(idlist.GetNumberOfIds())]

        length = weights.GetMaxId() + 1
        arr = np.zeros(length)
        for i in range(length):
            arr[i] = weights.GetTuple(i)[0]

        poly = dijkstra.GetOutput()

        vdata = numpy2vtk(arr)
        vdata.SetName("CumulativeWeights")
        poly.GetPointData().AddArray(vdata)

        vdata2 = numpy2vtk(ids, dtype=np.uint)
        vdata2.SetName("VertexIDs")
        poly.GetPointData().AddArray(vdata2)
        poly.GetPointData().Modified()

        dmesh = Mesh(poly, c='k')
        prop = vtk.vtkProperty()
        prop.DeepCopy(self.property)
        prop.SetLineWidth(3)
        prop.SetOpacity(1)
        dmesh.SetProperty(prop)
        dmesh.property = prop
        dmesh.name = "geodesicLine"
        return dmesh



