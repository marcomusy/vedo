from __future__ import division, print_function

import numpy as np
import os
import vtk
import vtkplotter.colors as colors
import vtkplotter.docs as docs
import vtkplotter.settings as settings
import vtkplotter.utils as utils

from vtkplotter.base import ActorBase

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

__doc__ = (
    """
Submodule extending the ``vtkActor`` object functionality.
"""
    + docs._defs
)

__all__ = ["Mesh", "merge"]


####################################################
def Actor(*args, **kargs):
    """``Actor`` class is obsolete: use ``Mesh`` instead, with same syntax."""
    colors.printc("WARNING: Actor() is obsolete, use Mesh() instead, with same syntax.", box='=', c=1)
    #raise RuntimeError()
    return Mesh(*args, **kargs)

def merge(*meshs):
    """
    Build a new mesh formed by the fusion of the polygonal meshes of the input objects.
    Similar to Assembly, but in this case the input objects become a single mesh.

    .. hint:: |thinplate_grid.py|_ |value-iteration.py|_

        |thinplate_grid| |value-iteration|
    """
    acts = []
    for a in utils.flatten(meshs):
        if isinstance(a, vtk.vtkAssembly):
            acts += a.getMeshes()
        elif a:
            acts += [a]

    if len(acts) == 1:
        return acts[0].clone()
    elif len(acts) == 0:
        return None

    polylns = vtk.vtkAppendPolyData()
    for a in acts:
        polylns.AddInputData(a.polydata())
    polylns.Update()
    pd = polylns.GetOutput()
    return Mesh(pd)


####################################################
class Mesh(vtk.vtkFollower, ActorBase):
    """
    Build an instance of object ``Mesh`` derived from ``vtkActor``.

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
        vtk.vtkActor.__init__(self)
        ActorBase.__init__(self)

        self._polydata = None
        self._mapper = vtk.vtkPolyDataMapper()

        self._mapper.SetInterpolateScalarsBeforeMapping(settings.interpolateScalarsBeforeMapping)

        if settings.usePolygonOffset:
            self._mapper.SetResolveCoincidentTopologyToPolygonOffset()
            pof, pou = settings.polygonOffsetFactor, settings.polygonOffsetUnits
            self._mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(pof, pou)

        inputtype = str(type(inputobj))
        # print('inputtype',inputtype)

        if inputobj is None:
            self._polydata = vtk.vtkPolyData()
        elif isinstance(inputobj, Mesh) or isinstance(inputobj, vtk.vtkActor):
            polyCopy = vtk.vtkPolyData()
            polyCopy.DeepCopy(inputobj.GetMapper().GetInput())
            self._polydata = polyCopy
            self._mapper.SetInputData(polyCopy)
            self._mapper.SetScalarVisibility(inputobj.GetMapper().GetScalarVisibility())
            pr = vtk.vtkProperty()
            pr.DeepCopy(inputobj.GetProperty())
            self.SetProperty(pr)
        elif "PolyData" in inputtype:
            if inputobj.GetNumberOfCells() == 0:
                carr = vtk.vtkCellArray()
                for i in range(inputobj.GetNumberOfPoints()):
                    carr.InsertNextCell(1)
                    carr.InsertCellPoint(i)
                inputobj.SetVerts(carr)
            self._polydata = inputobj  # cache vtkPolyData and mapper for speed
        elif "structured" in inputtype.lower() or "RectilinearGrid" in inputtype:
            if settings.visibleGridEdges:
                gf = vtk.vtkExtractEdges()
                gf.SetInputData(inputobj)
            else:
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(inputobj)
            gf.Update()
            self._polydata = gf.GetOutput()
        elif "trimesh" in inputtype:
            tact = utils.trimesh2vtk(inputobj, alphaPerCell=False)
            self._polydata = tact.polydata()
        elif "meshio" in inputtype:
            if inputobj.cells: # assume [vertices, faces]
                mcells =[]
                if 'triangle' in inputobj.cells.keys():
                    mcells += inputobj.cells['triangle'].tolist()
                if 'quad' in inputobj.cells.keys():
                    mcells += inputobj.cells['quad'].tolist()
                self._polydata = utils.buildPolyData(inputobj.points, mcells)
            else:
                self._polydata = utils.buildPolyData(inputobj.points, None)
            if inputobj.point_data:
                vptdata = numpy_to_vtk(inputobj.point_data, deep=True)
                self._polydata.SetPointData(vptdata)
            if inputobj.cell_data:
                vcldata = numpy_to_vtk(inputobj.cell_data, deep=True)
                self._polydata.SetPointData(vcldata)
        elif utils.isSequence(inputobj):
            if len(inputobj) == 2: # assume [vertices, faces]
                self._polydata = utils.buildPolyData(inputobj[0], inputobj[1])
            else:
                self._polydata = utils.buildPolyData(inputobj, None)
        elif hasattr(inputobj, "GetOutput"): # passing vtk object
            if hasattr(inputobj, "Update"): inputobj.Update()
            self._polydata = inputobj.GetOutput()
        else:
            colors.printc("Error: cannot build mesh from type:\n", inputtype, c=1)
            raise RuntimeError()

        self.SetMapper(self._mapper)

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

            if self._mapper:
                self._mapper.SetInputData(self._polydata)

        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        self._bfprop = None  # backface property holder
        self._scals_idx = 0  # index of the active scalar changed from CLI
        self._ligthingnr = 0

        prp = self.GetProperty()
        prp.SetInterpolationToPhong()

        if settings.renderPointsAsSpheres:
            if hasattr(prp, 'RenderPointsAsSpheresOn'):
                prp.RenderPointsAsSpheresOn()

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
                                break

            if arrexists == False:
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
    def __add__(self, meshs):
        from vtkplotter.assembly import Assembly
        if isinstance(meshs, list):
            alist = [self]
            for l in meshs:
                if isinstance(l, vtk.vtkAssembly):
                    alist += l.getMeshes()
                else:
                    alist += l
            return Assembly(alist)
        elif isinstance(meshs, vtk.vtkAssembly):
            meshs.AddPart(self)
            return meshs
        return Assembly([self, meshs])

    def __str__(self):
        utils.printInfo(self)
        return ""

    def _update(self, polydata):
        """Overwrite the polygonal mesh with a new vtkPolyData."""
        self._polydata = polydata
        self._mapper.SetInputData(polydata)
        self._mapper.Modified()
        return self

    def getPoints(self, transformed=True, copy=False):
        """Obsolete, use points() instead."""
        colors.printc("WARNING: getPoints() is obsolete, use points() instead.", box='=', c=1)
        return self.points(transformed=transformed, copy=copy)

    def setPoints(self, pts):
        """Obsolete, use points(pts) instead."""
        colors.printc("WARNING: setPoints(pts) is obsolete, use points(pts) instead.", box='=', c=1)
        return self.points(pts)

    def coordinates(self, transformed=True, copy=False):
        """Obsolete, use points() instead."""
        colors.printc("WARNING: coordinates() is obsolete, use points() instead.", box='=', c=1)
        return self.points(transformed=transformed, copy=copy)

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
        if pts is None: # getter
            poly = self.polydata(transformed)
            if copy:
                return np.array(vtk_to_numpy(poly.GetPoints().GetData()))
            else:
                return vtk_to_numpy(poly.GetPoints().GetData())

        elif (utils.isSequence(pts) and not utils.isSequence(pts[0])) or isinstance(pts, (int, np.integer)):
            #passing a list of indices or a single index
            return vtk_to_numpy(self.polydata(transformed).GetPoints().GetData())[pts]

        else: # setter
            vpts = self._polydata.GetPoints()
            vpts.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
            self._polydata.GetPoints().Modified()
            # reset mesh to identity matrix position/rotation:
            self.PokeMatrix(vtk.vtkMatrix4x4())
            return self

    def faces(self):
        """Get cell polygonal connectivity ids as a python ``list``.
        The output format is: [[id0 ... idn], [id0 ... idm],  etc].
        """
        #Get cell connettivity ids as a 1D array. The vtk format is:
        #    [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        arr1d = vtk_to_numpy(self.polydata(False).GetPolys().GetData())
        if len(arr1d) == 0:
            arr1d = vtk_to_numpy(self.polydata(False).GetStrips().GetData())

        #conn = arr1d.reshape(ncells, int(len(arr1d)/len(arr1d)))
        #return conn[:, 1:]
        # instead of:

        i = 0
        conn = []
        n = len(arr1d)
        for idummy in range(n):
#            cell = []
#            for k in range(arr1d[i]):
#                cell.append(arr1d[i+k+1])
            cell = [arr1d[i+k+1] for k in range(arr1d[i])]
            conn.append(cell)
            i += arr1d[i]+1
            if i >= n:
                break
        return conn # cannot always make a numpy array of it!


    def addScalarBar(self,
                     pos=(0.8,0.05),
                     title="",
                     titleXOffset=0,
                     titleYOffset=15,
                     titleFontSize=12,
                     nlabels=10,
                     c=None,
                     horizontal=False,
                     vmin=None, vmax=None,
    ):
        """
        Add a 2D scalar bar to mesh.

        |mesh_bands| |mesh_bands.py|_
        """
        import vtkplotter.addons as addons
        self.scalarbar = addons.addScalarBar(self,
                 pos,
                 title,
                 titleXOffset,
                 titleYOffset,
                 titleFontSize,
                 nlabels,
                 c,
                 horizontal,
                 vmin, vmax,
                 )
        return self

    def addScalarBar3D(
        self,
        pos=(0, 0, 0),
        normal=(0, 0, 1),
        sx=0.1,
        sy=2,
        title='',
        titleXOffset = -1.4, # space btw title and scale
        titleYOffset = 0.0,
        titleSize =  1.5,
        titleRotation = 0.0,
        nlabels=9,
        precision=3,
        labelOffset = 0.4,  # space btw numeric labels and scale
        c=None,
        alpha=1,
        cmap=None,
    ):
        """
        Draw a 3D scalar bar to mesh.

        |mesh_coloring| |mesh_coloring.py|_
        """
        import vtkplotter.addons as addons
        self.scalarbar = addons.addScalarBar3D(self,
                                                pos,
                                                normal,
                                                sx, sy,
                                                title,
                                                titleXOffset,
                                                titleYOffset,
                                                titleSize,
                                                titleRotation,
                                                nlabels,
                                                precision,
                                                labelOffset,
                                                c, alpha, cmap)
        return self.scalarbar


    def texture(self, tname,
                tcoords=None,
                interpolate=True,
                repeat=True,
                edgeClamp=False,
                ):
        """Assign a texture to mesh from image file or predefined texture `tname`.
        If tname is ``None`` texture is disabled.

        :param bool interpolate: turn on/off linear interpolation of the texture map when rendering.
        :param bool repeat: repeat of the texture when tcoords extend beyond the [0,1] range.
        :param bool edgeClamp: turn on/off the clamping of the texture map when
            the texture coords extend beyond the [0,1] range.
            Only used when repeat is False, and edge clamping is supported by the graphics card.
        """
        pd = self.polydata(False)
        if tname is None:
            pd.GetPointData().SetTCoords(None)
            pd.GetPointData().Modified()
            return self

        if isinstance(tname, vtk.vtkTexture):
            tu = tname
        else:
            if tcoords is not None:
                if not isinstance(tcoords, np.ndarray):
                    tcoords = np.array(tcoords)
                if tcoords.ndim != 2:
                    colors.printc('tcoords must be a 2-dimensional array', c=1)
                    return self
                if tcoords.shape[0] != pd.GetNumberOfPoints():
                    colors.printc('Error in texture(): nr of texture coords must match nr of points', c=1)
                    return self
                if tcoords.shape[1] != 2:
                    colors.printc('Error in texture(): vector must have 2 components', c=1)
                tarr = numpy_to_vtk(tcoords)
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
                    pd.GetPointData().SetTCoords(tc)
                    pd.GetPointData().Modified()

            fn = settings.textures_path + tname + ".jpg"
            if os.path.exists(tname):
                fn = tname
            elif not os.path.exists(fn):
                colors.printc("~sad File does not exist or texture", tname,
                              "not found in", settings.textures_path, c="r")
                colors.printc("~pin Available built-in textures:", c="m", end=" ")
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


    def deletePoints(self, indices, renamePoints=False):
        """Delete a list of vertices identified by their index.

        :param bool renamePoints: if True, point indices and faces are renamed.
            If False, vertices are not really deleted and faces indices will
            stay unchanged (default, faster).

        |deleteMeshPoints| |deleteMeshPoints.py|_
        """
        cellIds = vtk.vtkIdList()
        self._polydata.BuildLinks()
        for i in indices:
            self._polydata.GetPointCells(i, cellIds)
            for j in range(cellIds.GetNumberOfIds()):
                self._polydata.DeleteCell(cellIds.GetId(j))  # flag cell

        self._polydata.RemoveDeletedCells()

        if renamePoints:
            coords = self.points(transformed=False)
            faces = self.faces()
            pts_inds = np.unique(faces) # flattened array

            newfaces = []
            for f in faces:
                newface=[]
                for i in f:
                    idx = np.where(pts_inds==i)[0][0]
                    newface.append(idx)
                newfaces.append(newface)

            newpoly = utils.buildPolyData(coords[pts_inds], newfaces)
            return self._update(newpoly)

        self._mapper.Modified()
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

    def alpha(self, opacity=None):
        """Set/get mesh's transparency. Same as `mesh.opacity()`."""
        if opacity is None:
            return self.GetProperty().GetOpacity()

        self.GetProperty().SetOpacity(opacity)
        bfp = self.GetBackfaceProperty()
        if bfp:
            if opacity < 1:
                self._bfprop = bfp
                self.SetBackfaceProperty(None)
            else:
                self.SetBackfaceProperty(self._bfprop)
        return self

    def opacity(self, alpha=None):
        """Set/get mesh's transparency. Same as `mesh.alpha()`."""
        return self.alpha(alpha)

    def wireframe(self, value=True):
        """Set mesh's representation as wireframe or solid surface.
        Same as `wireframe()`."""
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

    def gouraud(self):
        """Set surface interpolation to Gouraud."""
        self.GetProperty().SetInterpolationToGouraud()
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

    def pointSize(self, ps=None):
        """Set/get mesh's point size of vertices. Same as `ps()`"""
        if ps is not None:
            if isinstance(self, vtk.vtkAssembly):
                cl = vtk.vtkPropCollection()
                self.GetActors(cl)
                cl.InitTraversal()
                a = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
                a.GetProperty().SetRepresentationToPoints()
                a.GetProperty().SetPointSize(ps)
            else:
                self.GetProperty().SetRepresentationToPoints()
                self.GetProperty().SetPointSize(ps)
        else:
            return self.GetProperty().GetPointSize()
        return self

    def ps(self, pointSize=None):
        """Set/get mesh's point size of vertices. Same as `pointSize()`"""
        return self.pointSize(pointSize)

    def color(self, c=False):
        """
        Set/get mesh's color.
        If None is passed as input, will use colors from active scalars.
        Same as `c()`.
        """
        if c is False:
            return np.array(self.GetProperty().GetColor())
        elif c is None:
            self._mapper.ScalarVisibilityOn()
            return self
        elif isinstance(c, str):
            if c in colors._mapscales_cmaps:
                self.cmap = c
                if self._polydata.GetPointData().GetScalars():
                    aname = self._polydata.GetPointData().GetScalars().GetName()
                    if aname: self.pointColors(aname, cmap=c)
                if self._polydata.GetCellData().GetScalars():
                    aname = self._polydata.GetCellData().GetScalars().GetName()
                    if aname: self.cellColors(aname, cmap=c)
                self._mapper.ScalarVisibilityOn()
                return self
        self._mapper.ScalarVisibilityOff()
        cc = colors.getColor(c)
        self.GetProperty().SetColor(cc)
        if self.trail:
            self.trail.GetProperty().SetColor(cc)
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
            colors.printc("~noentry backColor(): only active for alpha=1", c="y")
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

    def clean(self, tol=None):
        """
        Clean mesh polydata. Can also be used to decimate a mesh if ``tol`` is large.
        If ``tol=None`` only removes coincident points.

        :param tol: defines how far should be the points from each other
            in terms of fraction of the bounding box length.

        |moving_least_squares1D| |moving_least_squares1D.py|_

            |recosurface| |recosurface.py|_
        """
        poly = self.polydata(False)
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.PointMergingOn()
        cleanPolyData.ConvertLinesToPointsOn()
        cleanPolyData.ConvertPolysToLinesOn()
        cleanPolyData.SetInputData(poly)
        if tol:
            cleanPolyData.SetTolerance(tol)
        cleanPolyData.Update()
        return self._update(cleanPolyData.GetOutput())

    def quantize(self, binSize):
        """
        The user should input binSize and all {x,y,z} coordinates
        will be quantized to that absolute grain size.

        Example:
            .. code-block:: python

                from vtkplotter import Paraboloid
                Paraboloid().lw(0.1).quantize(0.1).show()
        """
        poly = self.polydata(False)
        qp = vtk.vtkQuantizePolyDataPoints()
        qp.SetInputData(poly)
        qp.SetQFactor(binSize)
        qp.Update()
        return self._update(qp.GetOutput())

    def xbounds(self):
        """Get the mesh bounds `[xmin,xmax]`."""
        b = self.polydata(True).GetBounds()
        return (b[0], b[1])

    def ybounds(self):
        """Get the mesh bounds `[ymin,ymax]`."""
        b = self.polydata(True).GetBounds()
        return (b[2], b[3])

    def zbounds(self):
        """Get the mesh bounds `[zmin,zmax]`."""
        b = self.polydata(True).GetBounds()
        return (b[4], b[5])

    def averageSize(self):
        """Calculate the average size of a mesh.
        This is the mean of the vertex distances from the center of mass."""
        cm = self.centerOfMass()
        coords = self.points(copy=False)
        if not len(coords):
            return 0
        s, c = 0.0, 0.0
        n = len(coords)
        step = int(n / 10000.0) + 1
        for i in np.arange(0, n, step):
            s += utils.mag(coords[i] - cm)
            c += 1
        return s / c

    def diagonalSize(self):
        """Get the length of the diagonal of mesh bounding box."""
        b = self.polydata().GetBounds()
        return np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)

    def maxBoundSize(self):
        """Get the maximum dimension in x, y or z of the mesh bounding box."""
        b = self.polydata(True).GetBounds()
        return max(abs(b[1] - b[0]), abs(b[3] - b[2]), abs(b[5] - b[4]))

    def centerOfMass(self):
        """Get the center of mass of mesh.

        |fatlimb| |fatlimb.py|_
        """
        cmf = vtk.vtkCenterOfMass()
        cmf.SetInputData(self.polydata(True))
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)

    def volume(self, value=None):
        """Get/set the volume occupied by mesh."""
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        v = mass.GetVolume()
        if value is not None:
            if not v:
                colors.printc("~bomb Volume is zero: cannot rescale.", c=1, end="")
                colors.printc(" Consider adding mesh.triangle()", c=1)
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
        mass.SetInputData(self.polydata())
        mass.Update()
        ar = mass.GetSurfaceArea()
        if value is not None:
            if not ar:
                colors.printc("~bomb Area is zero: cannot rescale.", c=1, end="")
                colors.printc(" Consider adding mesh.triangle()", c=1)
                return self
            self.scale(value / ar)
            return self
        else:
            return ar

    def closestPoint(self, pt, N=1, radius=None, returnIds=False):
        """
        Find the closest point(s) on a mesh given from the input point `pt`.

        :param int N: if greater than 1, return a list of N ordered closest points.
        :param float radius: if given, get all points within that radius.
        :param bool returnIds: return points IDs instead of point coordinates.

        .. hint:: |align1.py|_ |fitplanes.py|_  |quadratic_morphing.py|_

            |align1| |quadratic_morphing|

        .. note:: The appropriate kd-tree search locator is built on the
            fly and cached for speed.
        """
        poly = self.polydata(True)

        if N > 1 or radius:
            plocexists = self.point_locator
            if not plocexists or (plocexists and self.point_locator is None):
                point_locator = vtk.vtkPointLocator()
                point_locator.SetDataSet(poly)
                point_locator.BuildLocator()
                self.point_locator = point_locator

            vtklist = vtk.vtkIdList()
            if N > 1:
                self.point_locator.FindClosestNPoints(N, pt, vtklist)
            else:
                self.point_locator.FindPointsWithinRadius(radius, pt, vtklist)
            if returnIds:
                return [int(vtklist.GetId(k)) for k in range(vtklist.GetNumberOfIds())]
            else:
                trgp = []
                for i in range(vtklist.GetNumberOfIds()):
                    trgp_ = [0, 0, 0]
                    vi = vtklist.GetId(i)
                    poly.GetPoints().GetPoint(vi, trgp_)
                    trgp.append(trgp_)
                return np.array(trgp)

        clocexists = self.cell_locator
        if not clocexists or (clocexists and self.cell_locator is None):
            cell_locator = vtk.vtkCellLocator()
            cell_locator.SetDataSet(poly)
            cell_locator.BuildLocator()
            self.cell_locator = cell_locator

        trgp = [0, 0, 0]
        cid = vtk.mutable(0)
        dist2 = vtk.mutable(0)
        subid = vtk.mutable(0)
        self.cell_locator.FindClosestPoint(pt, trgp, cid, subid, dist2)
        if returnIds:
            return int(cid)
        else:
            return np.array(trgp)


    def findCellsWithin(self, xbounds=(), ybounds=(), zbounds=(), c=None):
        """
        Find cells that are within specified bounds.
        Setting a color will add a vtk array to colorize these cells.
        """
        if len(xbounds) == 6:
            bnds = xbounds
        else:
            bnds = list(self.bounds())
            if len(xbounds) == 2:
                bnds[0] = xbounds[0]
                bnds[1] = xbounds[1]
            if len(ybounds) == 2:
                bnds[2] = ybounds[0]
                bnds[3] = ybounds[1]
            if len(zbounds) == 2:
                bnds[4] = zbounds[0]
                bnds[5] = zbounds[1]

        cellIds = vtk.vtkIdList()
        self.cell_locator = vtk.vtkCellTreeLocator()
        self.cell_locator.SetDataSet(self.polydata())
        #self.cell_locator.SetNumberOfCellsPerNode(2)
        self.cell_locator.BuildLocator()
        self.cell_locator.FindCellsWithinBounds(bnds, cellIds)

        if c is not None:
            cellData = vtk.vtkUnsignedCharArray()
            cellData.SetNumberOfComponents(3)
            cellData.SetName('CellsWithinBoundsColor')
            cellData.SetNumberOfTuples(self.polydata(False).GetNumberOfCells())
            defcol = np.array(self.color())*255
            for i in range(cellData.GetNumberOfTuples()):
                cellData.InsertTuple(i, defcol)
            self.polydata(False).GetCellData().SetScalars(cellData)
            self._mapper.ScalarVisibilityOn()
            flagcol = np.array(colors.getColor(c))*255

        cids = []
        for i in range(cellIds.GetNumberOfIds()):
            cid = cellIds.GetId(i)
            if c is not None:
                cellData.InsertTuple(cid, flagcol)
            cids.append(cid)

        return np.array(cids)


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

    def clone(self, transformed=True):
        """
        Clone a ``Mesh(vtkActor)`` and make an exact copy of it.

        :param transformed: if `False` ignore any previous transformation applied to the mesh.

        |mirror| |mirror.py|_
        """
        poly = self.polydata(transformed=transformed)
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(poly)

        cloned = Mesh()
        cloned._polydata = polyCopy
        cloned._mapper.SetInputData(polyCopy)
        cloned._mapper.SetScalarVisibility(self._mapper.GetScalarVisibility())
        cloned._mapper.SetScalarRange(self._mapper.GetScalarRange())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.GetProperty())
        cloned.SetProperty(pr)
        cloned.base = self.base
        cloned.top = self.top
        if self.trail:
            n = len(self.trailPoints)
            cloned.addTrail(self.trailOffset, self.trailSegmentSize*n, n,
                            None, None, self.trail.GetProperty().GetLineWidth())
        if self.shadow:
            cloned.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                             self.shadow.GetProperty().GetColor(),
                             self.shadow.GetProperty().GetOpacity())
        return cloned

    def transformMesh(self, transformation):
        """
        Apply this transformation to the polygonal data,
        not to the mesh transformation, which is reset.

        :param transformation: ``vtkTransform`` or ``vtkMatrix4x4`` object.
        """
        if isinstance(transformation, vtk.vtkMatrix4x4):
            tr = vtk.vtkTransform()
            tr.SetMatrix(transformation)
            transformation = tr

        tf = vtk.vtkTransformPolyDataFilter()

        tf.SetTransform(transformation)
        tf.SetInputData(self.polydata())
        tf.Update()
        self.PokeMatrix(vtk.vtkMatrix4x4())  # identity
        return self._update(tf.GetOutput())

    def normalize(self):
        """
        Shift mesh center of mass at origin and scale its average size to unit.
        """
        cm = self.centerOfMass()
        coords = self.points()
        if not len(coords):
            return
        pts = coords - cm
        xyz2 = np.sum(pts * pts, axis=0)
        scale = 1 / np.sqrt(np.sum(xyz2) / len(pts))
        t = vtk.vtkTransform()
        t.Scale(scale, scale, scale)
        t.Translate(-cm)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(self._polydata)
        tf.SetTransform(t)
        tf.Update()
        return self._update(tf.GetOutput())

    def mirror(self, axis="x"):
        """
        Mirror the mesh  along one of the cartesian axes.

        .. note::  ``axis='n'``, will flip only mesh normals.

        |mirror| |mirror.py|_
        """
        poly = self.polydata(transformed=True)
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(poly)

        sx, sy, sz = 1, 1, 1
        dx, dy, dz = self.GetPosition()
        if axis.lower() == "x":
            sx = -1
        elif axis.lower() == "y":
            sy = -1
        elif axis.lower() == "z":
            sz = -1
        elif axis.lower() == "n":
            pass
        else:
            colors.printc("~times Error in mirror(): mirror must be set to x, y, z or n.",
                          c=1)
            raise RuntimeError()

        if axis != "n":
            for j in range(polyCopy.GetNumberOfPoints()):
                p = [0, 0, 0]
                polyCopy.GetPoint(j, p)
                polyCopy.GetPoints().SetPoint(
                    j,
                    p[0] * sx - dx * (sx - 1),
                    p[1] * sy - dy * (sy - 1),
                    p[2] * sz - dz * (sz - 1),
                )
        rs = vtk.vtkReverseSense()
        rs.SetInputData(polyCopy)
        rs.ReverseNormalsOn()
        rs.Update()
        polyCopy = rs.GetOutput()

        pdnorm = vtk.vtkPolyDataNormals()
        pdnorm.SetInputData(polyCopy)
        pdnorm.ComputePointNormalsOn()
        pdnorm.ComputeCellNormalsOn()
        pdnorm.FlipNormalsOff()
        pdnorm.ConsistencyOn()
        pdnorm.Update()
        return self._update(pdnorm.GetOutput())

    def flipNormals(self):
        """
        Flip all mesh normals. Same as `mesh.mirror('n')`.
        """
        return self.mirror("n")

    def shrink(self, fraction=0.85):
        """Shrink the triangle polydata in the representation of the input mesh.

        Example:
            .. code-block:: python

                from vtkplotter import *
                pot = load(datadir + 'teapot.vtk').shrink(0.75)
                s = Sphere(r=0.2).pos(0,0,-0.5)
                show(pot, s)

            |shrink| |shrink.py|_
        """
        poly = self.polydata(True)
        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputData(poly)
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
            colors.printc('~times Error in stretch(): Please define vectors', c='r')
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

    def crop(self, top=None, bottom=None, right=None, left=None, front=None, back=None):
        """Crop an ``Mesh`` object. Input object is modified.

        :param float top:    fraction to crop from the top plane (positive z)
        :param float bottom: fraction to crop from the bottom plane (negative z)
        :param float front:  fraction to crop from the front plane (positive y)
        :param float back:   fraction to crop from the back plane (negative y)
        :param float right:  fraction to crop from the right plane (positive x)
        :param float left:   fraction to crop from the left plane (negative x)

        Example:
            .. code-block:: python

                from vtkplotter import Sphere
                Sphere().crop(right=0.3, left=0.1).show()

            |cropped|
        """
        x0, x1, y0, y1, z0, z1 = self.GetBounds()

        cu = vtk.vtkBox()
        dx, dy, dz = x1-x0, y1-y0, z1-z0
        if top:    z1 = z1 - top*dz
        if bottom: z0 = z0 + bottom*dz
        if front:  y1 = y1 - front*dy
        if back:   y0 = y0 + back*dy
        if right:  x1 = x1 - right*dx
        if left:   x0 = x0 + left*dx
        cu.SetBounds(x0,x1, y0,y1, z0,z1)

        poly = self.polydata()
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        clipper.SetClipFunction(cu)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()

        self._update(clipper.GetOutput())
        return self

    def cutWithPlane(self, origin=(0, 0, 0), normal=(1, 0, 0), showcut=False):
        """
        Cut the mesh with the plane defined by a point and a normal.

        :param origin: the cutting plane goes through this point
        :param normal: normal of the cutting plane
        :param showcut: if `True` show the cut off part of the mesh as thin wireframe.

        :Example:
            .. code-block:: python

                from vtkplotter import Cube

                cube = Cube().cutWithPlane(normal=(1,1,1))
                cube.bc('pink').show()

            |cutcube|

        |trail| |trail.py|_
        """
        if str(normal) == "x":
            normal = (1, 0, 0)
        elif str(normal) == "y":
            normal = (0, 1, 0)
        elif str(normal) == "z":
            normal = (0, 0, 1)
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        self.computeNormals()
        poly = self.polydata()
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        clipper.SetClipFunction(plane)
        if showcut:
            clipper.GenerateClippedOutputOn()
        else:
            clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()

        self._update(clipper.GetOutput()).computeNormals()

        if showcut:
            from vtkplotter.assembly import Assembly
            c = self.GetProperty().GetColor()
            cpoly = clipper.GetClippedOutput()
            restmesh = Mesh(cpoly, c, 0.05).wireframe(True)
            restmesh.SetUserMatrix(self.GetMatrix())
            asse = Assembly([self, restmesh])
            self = asse
            return asse
        else:
            return self

    def cutWithMesh(self, mesh, invert=False):
        """
        Cut an ``Mesh`` mesh with another ``vtkPolyData`` or ``Mesh``.

        :param bool invert: if True return cut off part of mesh.

        .. hint:: |cutWithMesh.py|_ |cutAndCap.py|_

            |cutWithMesh| |cutAndCap|
        """
        if isinstance(mesh, vtk.vtkPolyData):
            polymesh = mesh
        elif isinstance(mesh, Mesh):
            polymesh = mesh.polydata()
        else:
            polymesh = mesh.GetMapper().GetInput()
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

        # add the SignedDistances to the grid
        poly.GetPointData().SetScalars(signedDistances)

        # use vtkClipDataSet to slice the grid with the polydata
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        clipper.SetInsideOut(not invert)
        clipper.SetValue(0.0)
        clipper.Update()
        return self._update(clipper.GetOutput())

    def cutWithPointLoop(self, points, invert=False):
        """
        Cut an ``Mesh`` object with a set of points forming a closed loop.
        """
        if isinstance(points, Mesh):
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
        return self._update(clipper.GetOutput())


    def cap(self, returnCap=False):
        """
        Generate a "cap" on a clipped mesh, or caps sharp edges.

        |cutAndCap| |cutAndCap.py|_
        """
        poly = self.polydata(True)

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
            return Mesh(tf.GetOutput())
        else:
            polyapp = vtk.vtkAppendPolyData()
            polyapp.AddInputData(poly)
            polyapp.AddInputData(tf.GetOutput())
            polyapp.Update()
            return self._update(polyapp.GetOutput()).clean()

    def threshold(self, scalars, vmin=None, vmax=None, useCells=False):
        """
        Extracts cells where scalar value satisfies threshold criterion.

        :param scalars: name of the scalars array.
        :type scalars: str, list
        :param float vmin: minimum value of the scalar
        :param float vmax: maximum value of the scalar
        :param bool useCells: if `True`, assume array scalars refers to cells.

        |mesh_threshold| |mesh_threshold.py|_
        """
        if utils.isSequence(scalars):
            self.addPointScalars(scalars, "threshold")
            scalars = "threshold"
        elif self.getPointArray(scalars) is None:
            colors.printc("~times No scalars found with name/nr:", scalars, c=1)
            raise RuntimeError()

        thres = vtk.vtkThreshold()
        thres.SetInputData(self._polydata)

        if useCells:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        thres.SetInputArrayToProcess(0, 0, 0, asso, scalars)

        if vmin is None and vmax is not None:
            thres.ThresholdByLower(vmax)
        elif vmax is None and vmin is not None:
            thres.ThresholdByUpper(vmin)
        else:
            thres.ThresholdBetween(vmin, vmax)
        thres.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(thres.GetOutput())
        gf.Update()
        return self._update(gf.GetOutput())

    def triangle(self, verts=True, lines=True):
        """
        Converts mesh polygons and strips to triangles.

        :param bool verts: if True, break input vertex cells into individual vertex cells
            (one point per cell). If False, the input vertex cells will be ignored.
        :param bool lines: if True, break input polylines into line segments.
            If False, input lines will be ignored and the output will have no lines.
        """
        tf = vtk.vtkTriangleFilter()
        tf.SetPassLines(lines)
        tf.SetPassVerts(verts)
        tf.SetInputData(self._polydata)
        tf.Update()
        return self._update(tf.GetOutput())

    def pointColors(self, scalars_or_colors, cmap="jet", alpha=1,
                    mode='scalars',
                    bands=None, vmin=None, vmax=None):
        """
        Set individual point colors by providing a list of scalar values and a color map.
        `scalars` can be a string name of the ``vtkArray``.

        if ``mode='colors'``, colorize vertices of a mesh one by one,
        passing a 1-to-1 list of colors.

        :param list alphas: single value or list of transparencies for each vertex

        :param cmap: color map scheme to transform a real number into a color.
        :type cmap: str, list, vtkLookupTable, matplotlib.colors.LinearSegmentedColormap
        :param alpha: mesh transparency. Can be a ``list`` of values one for each vertex.
        :type alpha: float, list
        :param int bands: group scalars in this number of bins, typically to form bands or stripes.
        :param float vmin: clip scalars to this minimum value
        :param float vmax: clip scalars to this maximum value

        .. hint::|mesh_coloring.py|_ |mesh_alphas.py|_ |mesh_bands.py|_ |mesh_custom.py|_

             |mesh_coloring| |mesh_alphas| |mesh_bands| |mesh_custom|
        """
        ####################################################################
        if 'color' in mode:
            return self._pointColors1By1(scalars_or_colors, alpha)
        ####################################################################

        poly = self.polydata(False)

        if isinstance(scalars_or_colors, str):  # if a name is passed
            scalars_or_colors = vtk_to_numpy(poly.GetPointData().GetArray(scalars_or_colors))

        try:
            n = len(scalars_or_colors)
        except TypeError:  # invalid type
            return self

        useAlpha = False
        if n != poly.GetNumberOfPoints():
            colors.printc('~times pointColors Error: nr. of scalars != nr. of points',
                          n, poly.GetNumberOfPoints(), c=1)
        if utils.isSequence(alpha):
            useAlpha = True
            if len(alpha) > n:
                colors.printc('~times pointColors Error: nr. of scalars < nr. of alpha values',
                              n, len(alpha), c=1)
                raise RuntimeError()
        if bands:
            scalars_or_colors = utils.makeBands(scalars_or_colors, bands)

        if vmin is None:
            vmin = np.min(scalars_or_colors)
        if vmax is None:
            vmax = np.max(scalars_or_colors)

        lut = vtk.vtkLookupTable()  # build the look-up table

        if utils.isSequence(cmap):
            sname = "pointColors_custom"
            lut.SetNumberOfTableValues(len(cmap))
            lut.Build()
            for i, c in enumerate(cmap):
                col = colors.getColor(c)
                r, g, b = col
                if useAlpha:
                    lut.SetTableValue(i, r, g, b, alpha[i])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        elif isinstance(cmap, vtk.vtkLookupTable):
            sname = "pointColors_lut"
            lut.DeepCopy(cmap)

        else:
            if isinstance(cmap, str):
                self.cmap = cmap
                sname = "pointColors_" + cmap
            else:
                sname = "pointColors"
            lut.SetNumberOfTableValues(256)
            lut.Build()
            for i in range(256):
                r, g, b = colors.colorMap(i, cmap, 0, 256)
                if useAlpha:
                    idx = int(i / 256 * len(alpha))
                    lut.SetTableValue(i, r, g, b, alpha[idx])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        arr = numpy_to_vtk(np.ascontiguousarray(scalars_or_colors), deep=True)
        arr.SetName(sname)
        self._mapper.SetArrayName(sname)
        if settings.autoResetScalarRange:
            self._mapper.SetScalarRange(vmin, vmax)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetScalarModeToUsePointData()
        self._mapper.ScalarVisibilityOn()
        poly.GetPointData().SetScalars(arr)
        poly.GetPointData().SetActiveScalars(sname)
        return self

    def _pointColors1By1(self, acolors, alphas=1):
        ptData = vtk.vtkUnsignedIntArray()
        ptData.SetName("VertexColors")
        lut = vtk.vtkLookupTable()
        n = self._polydata.GetNumberOfPoints()
        if len(acolors) != n or (utils.isSequence(alphas) and len(alphas) != n):
            colors.printc("~times _pointColors1By1(): mismatch in input list sizes.", c=1)
            return self
        lut.SetNumberOfTableValues(n)
        lut.Build()
        cols = colors.getColor(acolors)
        if not utils.isSequence(alphas):
            alphas = [alphas] * n
        for i in range(n):
            ptData.InsertNextValue(i)
            c = cols[i]
            lut.SetTableValue(i, c[0], c[1], c[2], alphas[i])
        self._polydata.GetPointData().SetScalars(ptData)
        self._polydata.GetPointData().Modified()
        self._mapper.SetScalarRange(0, n-1)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetArrayName("VertexColors")
        self._mapper.SetScalarModeToUsePointData()
        self._mapper.ScalarVisibilityOn()
        return self


    def cellColors(self, scalars_or_colors, cmap="jet", alpha=1, alphaPerCell=False,
                   mode='scalars',
                   bands=None, vmin=None, vmax=None):
        """
        Set individual cell colors by setting a list of scalars.

        If ``mode='scalars'`` (default), set individual cell colors by the
        provided list of scalars.

        If ``mode='colors'``, colorize the faces of a mesh one by one,
        passing a 1-to-1 list of colors and optionally a list of transparencies.

        :param alpha: mesh transparency. Can be a ``list`` of values one for each cell.
        :type alpha: float, list

        Only relevant with ``mode='scalars'`` (default):

        :param cmap: color map scheme to transform a real number into a color.
        :type cmap: str, list, vtkLookupTable, matplotlib.colors.LinearSegmentedColormap
        :param int bands: group scalars in this number of bins, typically to form bands of stripes.
        :param float vmin: clip scalars to this minimum value
        :param float vmax: clip scalars to this maximum value

        Only relevant with ``mode='colors'``:

        :param bool alphaPerCell: Only matters if `alpha` is a sequence. If so:
            if `True` assume that the list of opacities is independent
            on the colors (same color cells can have different opacity),
            this can be very slow for large meshes,

            if `False` [default] assume that the alpha matches the color list
            (same color has the same opacity).
            This is very fast even for large meshes.

        |mesh_coloring| |mesh_coloring.py|_
        """
        ####################################################################
        if 'color' in mode:
            return self._cellColors1By1(scalars_or_colors, alpha, alphaPerCell)
        ####################################################################

        poly = self.polydata(False)

        if isinstance(scalars_or_colors, str):  # if a name is passed
            scalars_or_colors = vtk_to_numpy(poly.GetCellData().GetArray(scalars_or_colors))

        n = len(scalars_or_colors)
        useAlpha = False
        if n != poly.GetNumberOfCells():
            colors.printc('~times cellColors(): nr. of scalars != nr. of cells',
                          n, poly.GetNumberOfCells(), c=1)
        if utils.isSequence(alpha):
            useAlpha = True
            if len(alpha) > n:
                colors.printc('~times cellColors(): nr. of scalars != nr. of alpha values',
                              n, len(alpha), c=1)
                raise RuntimeError()
        if bands:
            scalars_or_colors = utils.makeBands(scalars_or_colors, bands)

        if vmin is None:
            vmin = np.min(scalars_or_colors)
        if vmax is None:
            vmax = np.max(scalars_or_colors)

        lut = vtk.vtkLookupTable()  # build the look-up table

        if utils.isSequence(cmap):
            sname = "cellColors_custom"
            lut.SetNumberOfTableValues(len(cmap))
            lut.Build()
            for i, c in enumerate(cmap):
                col = colors.getColor(c)
                r, g, b = col
                if useAlpha:
                    lut.SetTableValue(i, r, g, b, alpha[i])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        elif isinstance(cmap, vtk.vtkLookupTable):
            sname = "cellColors_lut"
            lut.DeepCopy(cmap)

        else:
            if isinstance(cmap, str):
                self.cmap = cmap
                sname = "cellColors_" + cmap
            else:
                sname = "cellColors"
            lut.SetNumberOfTableValues(256)
            lut.Build()
            for i in range(256):
                r, g, b = colors.colorMap(i, cmap, 0, 256)
                if useAlpha:
                    idx = int(i / 256 * len(alpha))
                    lut.SetTableValue(i, r, g, b, alpha[idx])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        arr = numpy_to_vtk(np.ascontiguousarray(scalars_or_colors), deep=True)
        arr.SetName(sname)
        self._mapper.SetArrayName(sname)
        if settings.autoResetScalarRange:
            self._mapper.SetScalarRange(vmin, vmax)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetScalarModeToUseCellData()
        self._mapper.ScalarVisibilityOn()
        poly.GetCellData().SetScalars(arr)
        poly.GetCellData().SetActiveScalars(sname)
        return self

    def _cellColors1By1(self, acolors, alphas, alphaPerCell):
        cellData = vtk.vtkUnsignedIntArray()
        cellData.SetName("CellColors")

        n = self._polydata.GetNumberOfCells()
        if len(acolors) != n or (utils.isSequence(alphas) and len(alphas) != n):
            colors.printc("~times cellColors(): mismatch in input list sizes.",
                          len(acolors), n, c=1)
            return self

        lut = vtk.vtkLookupTable()

        if alphaPerCell:
            lut.SetNumberOfTableValues(n)
            lut.Build()
            cols = colors.getColor(acolors)
            if not utils.isSequence(alphas):
                alphas = [alphas] * n
            for i in range(n):
                cellData.InsertNextValue(i)
                c = cols[i]
                lut.SetTableValue(i, c[0], c[1], c[2], alphas[i])
        else:
            ucolors, uids, inds = np.unique(acolors, axis=0,
                                            return_index=True, return_inverse=True)
            nc = len(ucolors)

            if nc == 1:
                self.color(colors.getColor(ucolors[0]))
                if utils.isSequence(alphas):
                    self.alpha(alphas[0])
                else:
                    self.alpha(alphas)
                return self

            for i in range(n):
                cellData.InsertNextValue(int(inds[i]))

            lut.SetNumberOfTableValues(nc)
            lut.Build()

            cols = colors.getColor(ucolors)

            if not utils.isSequence(alphas):
                alphas = np.ones(n)

            for i in range(nc):
                c = cols[i]
                lut.SetTableValue(i, c[0], c[1], c[2], alphas[uids[i]])

        self._polydata.GetCellData().SetScalars(cellData)
        self._polydata.GetCellData().Modified()
        self._mapper.SetScalarRange(0, lut.GetNumberOfTableValues()-1)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetArrayName("CellColors")
        self._mapper.SetScalarModeToUseCellData()
        self._mapper.ScalarVisibilityOn()
        return self

    def addIDs(self, asfield=False):
        """
        Generate point and cell ids.

        :param bool asfield: flag to control whether to generate scalar or field data.
        """
        ids = vtk.vtkIdFilter()
        ids.SetInputData(self._polydata)
        ids.PointIdsOn()
        ids.CellIdsOn()
        if asfield:
            ids.FieldDataOn()
        else:
            ids.FieldDataOff()
        ids.Update()
        return self._update(ids.GetOutput())

    def addCurvatureScalars(self, method=0, lut=None):
        """
        Add scalars to ``Mesh`` that contains the
        curvature calculated in three different ways.

        :param int method: 0-gaussian, 1-mean, 2-max, 3-min curvature.
        :param lut: optional vtkLookUpTable up table.

        :Example:
            .. code-block:: python

                from vtkplotter import Torus
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

                from vtkplotter import Sphere

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

    def scalars(self, name_or_idx=None, datatype="point"):
        """Obsolete. Use methods getArrayNames(), getPointArray(), getCellArray(),
        addPointScalars(), addCellScalars or addPointVectors() instead."""
        colors.printc("WARNING: scalars() is obsolete!", c=1)
        colors.printc("       : Use getArrayNames(), getPointArray(), getCellArray(),", c=1)
        colors.printc("       : addPointScalars() or addPointVectors() instead.", c=1)
        #raise RuntimeError
        
        poly = self.polydata(False)

        # no argument: return list of available arrays
        if name_or_idx is None:
            ncd = poly.GetCellData().GetNumberOfArrays()
            npd = poly.GetPointData().GetNumberOfArrays()
            arrs = []
            for i in range(npd):
                #print(i, "PointData", poly.GetPointData().GetArrayName(i))
                arrs.append(["PointData", poly.GetPointData().GetArrayName(i)])
            for i in range(ncd):
                #print(i, "CellData", poly.GetCellData().GetArrayName(i))
                arrs.append(["CellData", poly.GetCellData().GetArrayName(i)])
            return arrs

        else:  # return a specific array (and set it as active one)

            pdata = poly.GetPointData()
            arr = None

            if 'point' in datatype.lower():
                if isinstance(name_or_idx, int):
                    name = pdata.GetArrayName(name_or_idx)
                else:
                    name = name_or_idx
                if name:
                    arr = pdata.GetArray(name)
                    data = pdata
                    self._mapper.SetScalarModeToUsePointData()


            if not arr or 'cell' in datatype.lower():
                cdata = poly.GetCellData()
                if isinstance(name_or_idx, int):
                    name = cdata.GetArrayName(name_or_idx)
                else:
                    name = name_or_idx
                if name:
                    arr = cdata.GetArray(name)
                    data = cdata
                    self._mapper.SetScalarModeToUseCellData()

            if arr:
                data.SetActiveScalars(name)
                self._mapper.ScalarVisibilityOn()
                if settings.autoResetScalarRange:
                    self._mapper.SetScalarRange(arr.GetRange())
                return vtk_to_numpy(arr)

            return None


    def subdivide(self, N=1, method=0):
        """Increase the number of vertices of a surface mesh.

        :param int N: number of subdivisions.
        :param int method: Loop(0), Linear(1), Adaptive(2), Butterfly(3)

        .. hint:: |tutorial_subdivide| |tutorial.py|_
        """
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputData(self.polydata())
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
            colors.printc("~times Error in subdivide: unknown method.", c="r")
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
        poly = self.polydata(True)
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

    def addGaussNoise(self, sigma):
        """
        Add gaussian noise.

        :param float sigma: sigma is expressed in percent of the diagonal size of mesh.

        :Example:
            .. code-block:: python

                from vtkplotter import Sphere

                Sphere().addGaussNoise(1.0).show()
        """
        sz = self.diagonalSize()
        pts = self.points()
        n = len(pts)
        ns = np.random.randn(n, 3) * sigma * sz / 100
        vpts = vtk.vtkPoints()
        vpts.SetNumberOfPoints(n)
        vpts.SetData(numpy_to_vtk(pts + ns, deep=True))
        self._polydata.SetPoints(vpts)
        self._polydata.GetPoints().Modified()
        self.addPointVectors(-ns, 'GaussNoise')
        return self

    def smoothLaplacian(self, niter=15, relaxfact=0.1, edgeAngle=15, featureAngle=60):
        """
        Adjust mesh point positions using `Laplacian` smoothing.

        :param int niter: number of iterations.
        :param float relaxfact: relaxation factor.
            Small `relaxfact` and large `niter` are more stable.
        :param float edgeAngle: edge angle to control smoothing along edges
            (either interior or boundary).
        :param float featureAngle: specifies the feature angle for sharp edge identification.

        .. hint:: |mesh_smoothers.py|_
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

        |mesh_smoothers| |mesh_smoothers.py|_
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

    def write(self, filename="mesh.vtk", binary=True):
        """Write mesh to file."""
        import vtkplotter.vtkio as vtkio

        return vtkio.write(self, filename, binary)

    def normalAt(self, i):
        """Return the normal vector at vertex point `i`."""
        normals = self.polydata(True).GetPointData().GetNormals()
        return np.array(normals.GetTuple(i))

    def normals(self, cells=False):
        """Retrieve vertex normals as a numpy array.

        :params bool cells: if `True` return cell normals.
        """
        if cells:
            vtknormals = self.polydata().GetCellData().GetNormals()
        else:
            vtknormals = self.polydata().GetPointData().GetNormals()
        if not vtknormals:
            self.computeNormals(cells=cells)
            if cells:
                vtknormals = self.polydata().GetCellData().GetNormals()
            else:
                vtknormals = self.polydata().GetPointData().GetNormals()

        return vtk_to_numpy(vtknormals)

    def polydata(self, transformed=True):
        """
        Returns the ``vtkPolyData`` object of a ``Mesh``.

        .. note:: If ``transformed=True`` returns a copy of polydata that corresponds
            to the current mesh's position in space.
        """
        if not transformed:
            if not self._polydata:
                self._polydata = self._mapper.GetInput()
            return self._polydata
        else:
            M = self.GetMatrix()
            if utils.isIdentity(M):
                # if identity return the original polydata
                if not self._polydata:
                    self._polydata = self._mapper.GetInput()
                return self._polydata
            else:
                # otherwise make a copy that corresponds to
                # the actual position in space of the mesh
                transform = vtk.vtkTransform()
                transform.SetMatrix(M)
                tp = vtk.vtkTransformPolyDataFilter()
                tp.SetTransform(transform)
                tp.SetInputData(self._polydata)
                tp.Update()
                return tp.GetOutput()

    def isInside(self, point, tol=0.0001):
        """
        Return True if point is inside a polydata closed surface.
        """
        poly = self.polydata(True)
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

    def insidePoints(self, pts, invert=False, tol=1e-05):
        """
        Return the sublist of points that are inside a polydata closed surface.

        |pca| |pca.py|_
        """
        poly = self.polydata(True)

        if isinstance(pts, Mesh):
            pointsPolydata = pts.polydata(True)
            pts = pts.points()
        else:
            vpoints = vtk.vtkPoints()
            vpoints.SetData(numpy_to_vtk(pts, deep=True))
            pointsPolydata = vtk.vtkPolyData()
            pointsPolydata.SetPoints(vpoints)

        sep = vtk.vtkSelectEnclosedPoints()
        sep.SetTolerance(tol)
        sep.SetInputData(pointsPolydata)
        sep.SetSurfaceData(poly)
        sep.Update()

        mask1, mask2 = [], []
        for i, p in enumerate(pts):
            if sep.IsInside(i):
                mask1.append(p)
            else:
                mask2.append(p)
        if invert:
            return mask2
        else:
            return mask1

    def cellCenters(self):
        """Get the list of cell centers of the mesh surface.

        |delaunay2d| |delaunay2d.py|_
        """
        vcen = vtk.vtkCellCenters()
        vcen.SetInputData(self.polydata(True))
        vcen.Update()
        return vtk_to_numpy(vcen.GetOutput().GetPoints().GetData())

    def boundaries(self, boundaryEdges=True, featureAngle=65, nonManifoldEdges=True):
        """
        Return a ``Mesh`` that shows the boundary lines of an input mesh.

        :param bool boundaryEdges: Turn on/off the extraction of boundary edges.
        :param float featureAngle: Specify the feature angle for extracting feature edges.
        :param bool nonManifoldEdges: Turn on/off the extraction of non-manifold edges.
        """
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(self.polydata())
        fe.SetBoundaryEdges(boundaryEdges)

        if featureAngle:
            fe.FeatureEdgesOn()
            fe.SetFeatureAngle(featureAngle)
        else:
            fe.FeatureEdgesOff()

        fe.SetNonManifoldEdges(nonManifoldEdges)
        fe.ColoringOff()
        fe.Update()
        return Mesh(fe.GetOutput(), c="p").lw(5)

    def connectedVertices(self, index, returnIds=False):
        """Find all vertices connected to an input vertex specified by its index.

        :param bool returnIds: return vertex IDs instead of vertex coordinates.

        |connVtx| |connVtx.py|_
        """
        mesh = self.polydata()

        cellIdList = vtk.vtkIdList()
        mesh.GetPointCells(index, cellIdList)

        idxs = []
        for i in range(cellIdList.GetNumberOfIds()):
            pointIdList = vtk.vtkIdList()
            mesh.GetCellPoints(cellIdList.GetId(i), pointIdList)
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
                mesh.GetPoints().GetPoint(i, p)
                trgp.append(p)
            return np.array(trgp)

    def connectedCells(self, index, returnIds=False):
        """Find all cellls connected to an input vertex specified by its index."""

        # Find all cells connected to point index
        dpoly = self.polydata()
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

                from vtkplotter import *
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

        intersectPoints = vtk.vtkPoints()
        self.line_locator.IntersectWithLine(p0, p1, intersectPoints, None)
        pts = []
        for i in range(intersectPoints.GetNumberOfPoints()):
            intersection = [0, 0, 0]
            intersectPoints.GetPoint(i, intersection)
            pts.append(intersection)
        return pts

    def projectOnPlane(self, direction='z'):
        """
        Project the mesh on one of the Cartesian planes.
        """
        coords = self.points(transformed=True)
        if 'z' == direction:
            coords[:, 2] = self.GetOrigin()[2]
            self.z(self.zbounds()[0])
        elif 'x' == direction:
            coords[:, 0] = self.GetOrigin()[0]
            self.x(self.xbounds()[0])
        elif 'y' == direction:
            coords[:, 1] = self.GetOrigin()[1]
            self.y(self.ybounds()[0])
        else:
            colors.printc("~times Error in projectOnPlane(): unknown direction", direction, c=1)
            raise RuntimeError()
        self.alpha(0.1).polydata(False).GetPoints().Modified()
        return self

    def silhouette(self, direction=None, borderEdges=True, featureAngle=None):
        """
        Return a new line ``Mesh`` which corresponds to the outer `silhouette`
        of the input as seen along a specified `direction`, this can also be
        a ``vtkCamera`` object.

        :param list direction: viewpoint direction vector.
            If *None* this is guessed by looking at the minimum
            of the sides of the bounding box.
        :param bool borderEdges: enable or disable generation of border edges
        :param float borderEdges: minimal angle for sharp edges detection.
            If set to `False` the functionality is disabled.

        |silhouette| |silhouette.py|_
        """
        sil = vtk.vtkPolyDataSilhouette()
        sil.SetInputData(self.polydata())
        if direction is None:
            b = self.GetBounds()
            i = np.argmin([b[1]-b[0], b[3]-b[2], b[5]-b[4]])
            d = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            sil.SetVector(d[i])
            sil.SetDirectionToSpecifiedVector()
        elif isinstance(direction, vtk.vtkCamera):
            sil.SetCamera(direction)
        else:
            sil.SetVector(direction)
            sil.SetDirectionToSpecifiedVector()
        if featureAngle is not None:
            sil.SetEnableFeatureAngle(1)
            sil.SetFeatureAngle(featureAngle)
            if featureAngle == False:
                sil.SetEnableFeatureAngle(0)

        sil.SetBorderEdges(borderEdges)
        sil.Update()
        return Mesh(sil.GetOutput()).lw(2).c('k')


    def addShadow(self, x=None, y=None, z=None, c=(0.5, 0.5, 0.5), alpha=1):
        """
        Generate a shadow out of an ``Mesh`` on one of the three Cartesian planes.
        The output is a new ``Mesh`` representing the shadow.
        This new mesh is accessible through `mesh.shadow`.
        By default the shadow mesh is placed on the bottom/back wall of the bounding box.

        :param float x,y,z: identify the plane to cast the shadow to ['x', 'y' or 'z'].
            The shadow will lay on the orthogonal plane to the specified axis at the
            specified value of either x, y or z.

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
        shad.c(c).alpha(alpha).wireframe(False)
        shad.flat().backFaceCulling()
        shad.GetProperty().LightingOff()
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

    def addTrail(self, offset=None, maxlength=None, n=50, c=None, alpha=None, lw=2):
        """Add a trailing line to mesh.
        This new mesh is accessible through `mesh.trail`.

        :param offset: set an offset vector from the object center.
        :param maxlength: length of trailing line in absolute units
        :param n: number of segments to control precision
        :param lw: line width of the trail

        .. hint:: See examples: |trail.py|_  |airplanes.py|_

            |trail|
        """
        if maxlength is None:
            maxlength = self.diagonalSize() * 20
            if maxlength == 0:
                maxlength = 1

        if self.trail is None:
            from vtkplotter.mesh import Mesh
            pos = self.GetPosition()
            self.trailPoints = [None] * n
            self.trailSegmentSize = maxlength / n
            self.trailOffset = offset

            ppoints = vtk.vtkPoints()  # Generate the polyline
            poly = vtk.vtkPolyData()
            ppoints.SetData(numpy_to_vtk([pos] * n))
            poly.SetPoints(ppoints)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(n)
            for i in range(n):
                lines.InsertCellPoint(i)
            poly.SetPoints(ppoints)
            poly.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper()

            if c is None:
                if hasattr(self, "GetProperty"):
                    col = self.GetProperty().GetColor()
                else:
                    col = (0.1, 0.1, 0.1)
            else:
                col = colors.getColor(c)

            if alpha is None:
                alpha = 1
                if hasattr(self, "GetProperty"):
                    alpha = self.GetProperty().GetOpacity()

            mapper.SetInputData(poly)
            tline = Mesh(poly, c=col, alpha=alpha)
            tline.SetMapper(mapper)
            tline.GetProperty().SetLineWidth(lw)
            self.trail = tline  # holds the vtkActor
        return self

    def updateTrail(self):
        currentpos = np.array(self.GetPosition())
        if self.trailOffset:
            currentpos += self.trailOffset
        lastpos = self.trailPoints[-1]
        if lastpos is None:  # reset list
            self.trailPoints = [currentpos] * len(self.trailPoints)
            return
        if np.linalg.norm(currentpos - lastpos) < self.trailSegmentSize:
            return

        self.trailPoints.append(currentpos)  # cycle
        self.trailPoints.pop(0)

        tpoly = self.trail.polydata()
        tpoly.GetPoints().SetData(numpy_to_vtk(self.trailPoints))
        return self


    def followCamera(self, cam=None):
        """Mesh object will follow camera movements and stay locked to it.

        :param vtkCamera cam: if `None` the text will auto-orient itself to the active camera.
            A ``vtkCamera`` object can also be passed.
        """
        if cam is False:
            self.SetCamera(None)
            return self
        if isinstance(cam, vtk.vtkCamera):
            self.SetCamera(cam)
        else:
            if not settings.plotter_instance or not settings.plotter_instance.camera:
                colors.printc("Error in followCamera(): needs an already rendered scene,", c=1)
                colors.printc("                         or passing a vtkCamera object.", c=1)
                return self
            self.SetCamera(settings.plotter_instance.camera)
        return self


    def isolines(self, n=10, vmin=None, vmax=None):
        """
        Return a new ``Mesh`` representing the isolines of the active scalars.

        :param int n: number of isolines in the range
        :param float vmin: minimum of the range
        :param float vmax: maximum of the range

        |isolines| |isolines.py|_
        """
        bcf = vtk.vtkBandedPolyDataContourFilter()
        bcf.SetInputData(self.polydata())
        bcf.SetScalarModeToValue()
        bcf.GenerateContourEdgesOn()
        r0, r1 = self._polydata.GetScalarRange()
        if vmin is None:
            vmin = r0
        if vmax is None:
            vmax = r1
        bcf.GenerateValues(n, vmin, vmax)
        bcf.Update()
        zpoly = bcf.GetContourEdgesOutput()
        zbandsact = Mesh(zpoly, c="k")
        zbandsact.GetProperty().SetLineWidth(1.5)
        return zbandsact

