#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from deprecated import deprecated

import vtkmodules.all as vtk
import vedo
from vedo.colors import color_map
from vedo.colors import get_color
from vedo.pointcloud import Points
from vedo.utils import buildPolyData
from vedo.utils import flatten, is_sequence, mag, mag2
from vedo.utils import numpy2vtk, vtk2numpy

__doc__ = """
Submodule to work with polygonal meshes

.. image:: https://vedo.embl.es/images/advanced/mesh_smoother2.png
"""

__all__ = ["Mesh", "merge"]


####################################################
def merge(*meshs, flag=False):
    """
    Build a new mesh formed by the fusion of the input polygonal Meshes (or Points).

    Similar to Assembly, but in this case the input objects become a single mesh entity.

    To keep track of the original identities of the input mesh you can use ``flag``.
    In this case a point array of IDs is added to the merged output mesh.

    .. hint:: warp1.py, value_iteration.py
        .. image:: https://vedo.embl.es/images/advanced/warp1.png
    """
    acts = [a for a in flatten(meshs) if a]

    if not acts:
        return None

    idarr = []
    polyapp = vtk.vtkAppendPolyData()
    for i, a in enumerate(acts):
        try:
            poly = a.polydata()
        except AttributeError:
            # so a vtkPolydata can also be passed
            poly = a
        polyapp.AddInputData(poly)
        if flag:
            idarr += [i] * poly.GetNumberOfPoints()
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

    E.g.:
        `Mesh( [ [[x1,y1,z1],[x2,y2,z2], ...],  [[0,1,2], [1,2,3], ...] ] )`

    Parameters
    ----------
    c : color
        color in RGB format, hex, symbol or name

    alpha : float
        mesh opacity [0,1]

    .. hint:: buildmesh.py (and many others!)
        ... image:: https://vedo.embl.es/images/basic/buildmesh.png
    """

    def __init__(
        self,
        inputobj=None,
        c=None,
        alpha=1,
    ):
        Points.__init__(self)

        self.line_locator = None

        self._mapper.SetInterpolateScalarsBeforeMapping(
            vedo.settings.interpolateScalarsBeforeMapping
        )

        if vedo.settings.usePolygonOffset:
            self._mapper.SetResolveCoincidentTopologyToPolygonOffset()
            pof, pou = (
                vedo.settings.polygonOffsetFactor,
                vedo.settings.polygonOffsetUnits,
            )
            self._mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(pof, pou)
            # self._mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(pof, pou)
            # self._mapper.SetRelativeCoincidentTopologyLineOffsetParameters(pof, pou)

            # a = vtk.reference(0)
            # b = vtk.reference(0)
            # mapper.GetResolveCoincidentTopologyPolygonOffsetParameters(a,b)
            # mapper.GetRelativeCoincidentTopologyPolygonOffsetParameters(a,b)
            # print(a,b)

        inputtype = str(type(inputobj))

        if inputobj is None:
            pass

        elif isinstance(inputobj, (Mesh, vtk.vtkActor)):
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
            tact = vedo.utils.trimesh2vedo(inputobj)
            self._data = tact.polydata()

        elif "meshio" in inputtype:  # meshio-4.0.11
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
            self._data = vedo.utils.meshlab2vedo(inputobj)

        elif is_sequence(inputobj):
            ninp = len(inputobj)
            if ninp == 0:
                self._data = vtk.vtkPolyData()
            elif ninp == 2:  # assume [vertices, faces]
                self._data = buildPolyData(inputobj[0], inputobj[1])
            else:  # assume [vertices] or vertices
                self._data = buildPolyData(inputobj, None)

        elif hasattr(inputobj, "GetOutput"):  # passing vtk object
            if hasattr(inputobj, "Update"):
                inputobj.Update()
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
                self._data = dataset.tomesh().polydata(False)
            else:
                self._data = dataset.polydata(False)

        else:
            try:
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(inputobj)
                gf.Update()
                self._data = gf.GetOutput()
            except:
                vedo.logger.error(f"cannot build mesh from type {inputtype}")
                raise RuntimeError()

        self._mapper.SetInputData(self._data)

        self.property = self.GetProperty()
        self.property.SetInterpolationToPhong()

        # set the color by c or by scalar
        if self._data:

            arrexists = False

            if c is None:
                ptdata = self._data.GetPointData()
                cldata = self._data.GetCellData()
                exclude = ["normals", "tcoord"]

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
                                break  # stop at first good one

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
                                break  # stop at first good one

            if not arrexists:
                if c is None:
                    c = "gold"
                    c = get_color(c)
                elif isinstance(c, float) and c <= 1:
                    c = color_map(c, "rainbow", 0, 1)
                else:
                    c = get_color(c)
                self.property.SetColor(c)
                self.property.SetAmbient(0.1)
                self.property.SetDiffuse(1)
                self.property.SetSpecular(0.05)
                self.property.SetSpecularPower(5)
                self._mapper.ScalarVisibilityOff()

        if alpha is not None:
            self.property.SetOpacity(alpha)


    def faces(self):
        """
        Get cell polygonal connectivity ids as a python `list`.
        The output format is: [[id0 ... idn], [id0 ... idm],  etc].
        """
        arr1d = vtk2numpy(self._data.GetPolys().GetData())
        if arr1d is None:
            return []

        # Get cell connettivity ids as a 1D array. vtk format is:
        # [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        if len(arr1d) == 0:
            arr1d = vtk2numpy(self._data.GetStrips().GetData())
            if arr1d is None:
                return []

        i = 0
        conn = []
        n = len(arr1d)
        if n:
            while True:
                cell = [arr1d[i + k] for k in range(1, arr1d[i] + 1)]
                conn.append(cell)
                i += arr1d[i] + 1
                if i >= n:
                    break
        return conn  # cannot always make a numpy array of it!

    def cells(self):
        """Alias for ``faces()``."""
        return self.faces()

    def lines(self, flat=False):
        """
        Get lines connectivity ids as a numpy array.
        Default format is [[id0,id1], [id3,id4], ...]

        Parameters
        ----------
        flat : bool
            return a 1D numpy array as e.g. [2, 10,20, 3, 10,11,12, 2, 70,80, ...]
        """
        # Get cell connettivity ids as a 1D array. The vtk format is:
        #    [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        arr1d = vtk2numpy(self.polydata(False).GetLines().GetData())

        if arr1d is None:
            return []

        if flat:
            return arr1d

        i = 0
        conn = []
        n = len(arr1d)
        for _ in range(n):
            cell = [arr1d[i + k + 1] for k in range(arr1d[i])]
            conn.append(cell)
            i += arr1d[i] + 1
            if i >= n:
                break

        return conn  # cannot always make a numpy array of it!

    def edges(self):
        """Return an array containing the edges connectivity."""
        extractEdges = vtk.vtkExtractEdges()
        extractEdges.SetInputData(self._data)
        # eed.UseAllPointsOn()
        extractEdges.Update()
        lpoly = extractEdges.GetOutput()

        arr1d = vtk2numpy(lpoly.GetLines().GetData())
        # [nids1, id0 ... idn, niids2, id0 ... idm,  etc].

        i = 0
        conn = []
        n = len(arr1d)
        for _ in range(n):
            cell = [arr1d[i + k + 1] for k in range(arr1d[i])]
            conn.append(cell)
            i += arr1d[i] + 1
            if i >= n:
                break
        return conn  # cannot always make a numpy array of it!

    def texture(
        self,
        tname,
        tcoords=None,
        interpolate=True,
        repeat=True,
        edge_clamp=False,
        scale=None,
        ushift=None,
        vshift=None,
        seam_threshold=None,
    ):
        """
        Assign a texture to mesh from image file or predefined texture `tname`.
        If tname is set to ``None`` texture is disabled.
        Input tname can also be an array or a `vtkTexture`.

        Parameters
        ----------
        tname : numpy.array, str, Picture, vtkTexture, None
            the inpout texture to be applied. Can be a numpy array, a path to an image file,
            a vedo Picture. The None value disables texture.

        tcoords : numpy.array, str
            this is the (u,v) texture coordinate array. Can also be a string of an existing array
            in the mesh.

        interpolate : bool, optional
            turn on/off linear interpolation of the texture map when rendering.

        repeat : bool, optional
            repeat of the texture when tcoords extend beyond the [0,1] range.

        edge_clamp : bool, optional
            turn on/off the clamping of the texture map when
            the texture coords extend beyond the [0,1] range.
            Only used when repeat is False, and edge clamping is supported by the graphics card.

        scale : bool, optional
            scale the texture image by this factor

        ushift : bool, optional
            shift u-coordinates of texture by this amount

        vshift : bool, optional
            shift v-coordinates of texture by this amount

        seam_threshold : float, optional
            try to seal seams in texture by collapsing triangles
            (test values around 1.0, lower values = stronger collapse)

        .. hint:: texturecubes.py
            .. image:: https://vedo.embl.es/images/basic/texturecubes.png
        """
        pd = self.polydata(False)

        if tname is None:  # disable texture
            pd.GetPointData().SetTCoords(None)
            pd.GetPointData().Modified()
            return self  ######################################

        if isinstance(tname, vtk.vtkTexture):
            tu = tname
            outimg = self._data

        elif isinstance(tname, vedo.Picture):
            tu = vtk.vtkTexture()
            outimg = tname.inputdata()

        elif is_sequence(tname):
            tu = vtk.vtkTexture()
            outimg = vedo.picture._get_img(tname)

        elif isinstance(tname, str):
            tu = vtk.vtkTexture()

            if "https://" in tname:
                try:
                    tname = vedo.io.download(tname, verbose=False)
                except:
                    vedo.logger.error(f"texture {tname} could not be downloaded")
                    return self

            fn = tname + ".jpg"
            if os.path.exists(tname):
                fn = tname
            else:
                vedo.logger.error(f"texture file {tname} does not exist")
                return self

            fnl = fn.lower()
            if ".jpg" in fnl or ".jpeg" in fnl:
                reader = vtk.vtkJPEGReader()
            elif ".png" in fnl:
                reader = vtk.vtkPNGReader()
            elif ".bmp" in fnl:
                reader = vtk.vtkBMPReader()
            else:
                vedo.logger.error(
                    "in texture() supported files are only PNG, BMP or JPG"
                )
                return self
            reader.SetFileName(fn)
            reader.Update()
            outimg = reader.GetOutput()

        else:
            vedo.logger.error(f"in texture() cannot understand input {type(tname)}")
            return self

        if tcoords is not None:

            if isinstance(tcoords, str):

                vtarr = pd.GetPointData().GetArray(tcoords)

            else:

                tcoords = np.asarray(tcoords)
                if tcoords.ndim != 2:
                    vedo.logger.error("tcoords must be a 2-dimensional array")
                    return self
                if tcoords.shape[0] != pd.GetNumberOfPoints():
                    vedo.logger.error("nr of texture coords must match nr of points")
                    return self
                if tcoords.shape[1] != 2:
                    vedo.logger.error("tcoords texture vector must have 2 components")
                vtarr = numpy2vtk(tcoords)
                vtarr.SetName("TCoordinates")

            pd.GetPointData().SetTCoords(vtarr)
            pd.GetPointData().Modified()

        elif not pd.GetPointData().GetTCoords():

            # TCoords still void..
            # check that there are no texture-like arrays:
            names = self.pointdata.keys()
            candidate_arr = ""
            for name in names:
                vtarr = pd.GetPointData().GetArray(name)
                if vtarr.GetNumberOfComponents() != 2:
                    continue
                t0, t1 = vtarr.GetRange()
                if t0 >= 0 and t1 <= 1:
                    candidate_arr = name

            if candidate_arr:

                vtarr = pd.GetPointData().GetArray(candidate_arr)
                pd.GetPointData().SetTCoords(vtarr)
                pd.GetPointData().Modified()

            else:
                # last resource is automatic mapping
                tmapper = vtk.vtkTextureMapToPlane()
                tmapper.AutomaticPlaneGenerationOn()
                tmapper.SetInputData(pd)
                tmapper.Update()
                tc = tmapper.GetOutput().GetPointData().GetTCoords()
                if scale or ushift or vshift:
                    ntc = vtk2numpy(tc)
                    if scale:
                        ntc *= scale
                    if ushift:
                        ntc[:, 0] += ushift
                    if vshift:
                        ntc[:, 1] += vshift
                    tc = numpy2vtk(tc)
                pd.GetPointData().SetTCoords(tc)
                pd.GetPointData().Modified()

        tu.SetInputData(outimg)
        tu.SetInterpolate(interpolate)
        tu.SetRepeat(repeat)
        tu.SetEdgeClamp(edge_clamp)

        self.property.SetColor(1, 1, 1)
        self._mapper.ScalarVisibilityOff()
        self.SetTexture(tu)

        if seam_threshold is not None:
            tname = self._data.GetPointData().GetTCoords().GetName()
            grad = self.gradient(tname)
            ugrad, vgrad = np.split(grad, 2, axis=1)
            ugradm, vgradm = vedo.utils.mag2(ugrad), vedo.utils.mag2(vgrad)
            gradm = np.log(ugradm + vgradm)
            largegrad_ids = np.arange(len(grad))[gradm > seam_threshold * 4]
            uvmap = self.pointdata[tname]
            # collapse triangles that have large gradient
            new_points = self.points(transformed=False)
            for f in self.faces():
                if np.isin(f, largegrad_ids).all():
                    id1, id2, id3 = f
                    uv1, uv2, uv3 = uvmap[f]
                    d12 = vedo.mag2(uv1 - uv2)
                    d23 = vedo.mag2(uv2 - uv3)
                    d31 = vedo.mag2(uv3 - uv1)
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

    @deprecated(reason=vedo.colors.red + "Please use compute_normals()" + vedo.colors.reset)
    def computeNormals(self, points=True, cells=True, featureAngle=None, consistency=True):
        "Deprecated. Please use compute_normals()"
        return self.compute_normals(points, cells, featureAngle, consistency)

    def compute_normals(self, points=True, cells=True, feature_angle=None, consistency=True):
        """
        Compute cell and vertex normals for the mesh.

        Parameters
        ----------
        points : bool
            do the computation for the vertices too

        cells : bool
            do the computation for the cells too

        feature_angle : float
            specify the angle that defines a sharp edge.
            If the difference in angle across neighboring polygons is greater than this value,
            the shared edge is considered "sharp" and it is splitted.

        consistency : bool
            turn on/off the enforcement of consistent polygon ordering.

        .. warning::
            if feature_angle is set to a float the Mesh can be modified, and it
            can have a different nr. of vertices from the original.
        """
        poly = self.polydata(False)
        pdnorm = vtk.vtkPolyDataNormals()
        pdnorm.SetInputData(poly)
        pdnorm.SetComputePointNormals(points)
        pdnorm.SetComputeCellNormals(cells)
        pdnorm.SetConsistency(consistency)
        pdnorm.FlipNormalsOff()
        if feature_angle:
            pdnorm.SetSplitting(True)
            pdnorm.SetFeatureAngle(feature_angle)
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
        If cell is a list of IDs only those cells will be reversed.

        - `normals=True` reverses the normals by multiplying the normal vector by -1
            (both point and cell normals, if present).
        """
        poly = self.polydata(False)

        if is_sequence(cells):
            for cell in cells:
                poly.ReverseCell(cell)
            poly.GetCellData().Modified()
            return self  ##############

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
        """Set mesh's representation as wireframe or solid surface."""
        if value:
            self.property.SetRepresentationToWireframe()
        else:
            self.property.SetRepresentationToSurface()
        return self

    def flat(self):
        """Set surface interpolation to Flat.

        .. image:: https://upload.wikimedia.org/wikipedia/commons/6/6b/Phong_components_version_4.png
        """
        self.property.SetInterpolationToFlat()
        return self

    def phong(self):
        """Set surface interpolation to Phong."""
        self.property.SetInterpolationToPhong()
        return self

    def backface_culling(self, value=True):
        """Set culling of polygons based on orientation
        of normal with respect to camera."""
        self.property.SetBackfaceCulling(value)
        return self

    def render_lines_as_tubes(self, value=True):
        """Wrap a fake tube around a simple line for visualization"""
        self.property.SetRenderLinesAsTubes(value)
        return self

    def frontface_culling(self, value=True):
        """Set culling of polygons based on orientation of normal with respect to camera."""
        self.property.SetFrontfaceCulling(value)
        return self

    @deprecated(reason=vedo.colors.red + "Please use backcolor()" + vedo.colors.reset)
    def backColor(self, bc=None):
        "Please use backcolor()"
        return self.backcolor(bc)

    def backcolor(self, bc=None):
        """
        Set/get mesh's backface color.
        """
        backProp = self.GetBackfaceProperty()

        if bc is None:
            if backProp:
                return backProp.GetDiffuseColor()
            return self

        if self.property.GetOpacity() < 1:
            return self

        if not backProp:
            backProp = vtk.vtkProperty()

        backProp.SetDiffuseColor(get_color(bc))
        backProp.SetOpacity(self.property.GetOpacity())
        self.SetBackfaceProperty(backProp)
        self._mapper.ScalarVisibilityOff()
        return self

    def bc(self, backcolor=False):
        """Shortcut for `mesh.backcolor()`."""
        return self.backcolor(backcolor)

    @deprecated(reason=vedo.colors.red + "Please use linewidth()" + vedo.colors.reset)
    def lineWidth(self, lw):
        "Deprecated: please use linewidth()"
        return self.linewidth(lw)

    def linewidth(self, lw=None):
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

    def lw(self, linewidth=None):
        """Set/get width of mesh edges. Same as `linewidth()`."""
        return self.linewidth(linewidth)

    def linecolor(self, lc=None):
        """Set/get color of mesh edges. Same as `lc()`."""
        if lc is None:
            return self.property.GetEdgeColor()
        else:
            self.property.EdgeVisibilityOn()
            self.property.SetEdgeColor(get_color(lc))
            return self

    def lc(self, linecolor=None):
        """Set/get color of mesh edges. Same as `linecolor()`."""
        return self.linecolor(linecolor)

    def volume(self):
        """Get/set the volume occupied by mesh."""
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        return mass.GetVolume()

    def area(self):
        """Get/set the surface area of mesh."""
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        return mass.GetSurfaceArea()

    def is_closed(self):
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

        .. hint:: shrink.py
            .. image:: https://vedo.embl.es/images/basic/shrink.png
        """
        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputData(self._data)
        shrink.SetShrinkFactor(fraction)
        shrink.Update()
        self.point_locator = None
        self.cell_locator = None
        return self._update(shrink.GetOutput())

    def stretch(self, q1, q2):
        """Stretch mesh between points `q1` and `q2`.

        .. hint:: aspring.py
            .. image:: https://vedo.embl.es/images/simulations/50738955-7e891800-11d9-11e9-85cd-02bd4f3f13ea.gif

        .. note:: for ``Mesh`` objects, two attributes ``mesh.base``,
            and ``mesh.top`` are always defined.
        """
        if self.base is None:
            vedo.logger.error(
                "in stretch() must define vectors mesh.base and mesh.top at creation"
            )
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
        return self

    def crop(
        self,
        top=None,
        bottom=None,
        right=None,
        left=None,
        front=None,
        back=None,
        bounds=None,
    ):
        """
        Crop an ``Mesh`` object.

        Parameters
        ----------
        top : float
            fraction to crop from the top plane (positive z)

        bottom : float
            fraction to crop from the bottom plane (negative z)

        front : float
            fraction to crop from the front plane (positive y)

        back : float
            fraction to crop from the back plane (negative y)

        right : float
            fraction to crop from the right plane (positive x)

        left : float
            fraction to crop from the left plane (negative x)

        bounds : list
            direct list of bounds passed as [x0,x1, y0,y1, z0,z1]

        Example:
            .. code-block:: python

                from vedo import Sphere
                Sphere().crop(right=0.3, left=0.1).show()

            .. image:: https://user-images.githubusercontent.com/32848391/57081955-0ef1e800-6cf6-11e9-99de-b45220939bc9.png
        """
        cu = vtk.vtkBox()
        x0, x1, y0, y1, z0, z1 = self.GetBounds()
        pos = np.array(self.GetPosition())
        x0, y0, z0 = [x0, y0, z0] - pos
        x1, y1, z1 = [x1, y1, z1] - pos

        if bounds is None:
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            if top:
                z1 = z1 - top * dz
            if bottom:
                z0 = z0 + bottom * dz
            if front:
                y1 = y1 - front * dy
            if back:
                y0 = y0 + back * dy
            if right:
                x1 = x1 - right * dx
            if left:
                x0 = x0 + left * dx
            bounds = (x0, x1, y0, y1, z0, z1)
        else:
            if bounds[0] is None:
                bounds[0] = x0
            if bounds[1] is None:
                bounds[1] = x1
            if bounds[2] is None:
                bounds[2] = y0
            if bounds[3] is None:
                bounds[3] = y1
            if bounds[4] is None:
                bounds[4] = z0
            if bounds[5] is None:
                bounds[5] = z1
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


    def cap(self, return_cap=False):
        """
        Generate a "cap" on a clipped mesh, or caps sharp edges.

        .. hint:: cutAndCap.py
            .. image:: https://vedo.embl.es/images/advanced/cutAndCap.png
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

        if return_cap:
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

        Parameters
        ----------
        polys : bool
            polygonal segments will be joined if they are contiguous

        reset : bool
            reset points ordering

        Warning:
            If triangle strips or polylines exist in the input data
            they will be passed through to the output data.
            This filter will only construct triangle strips if triangle polygons
            are available; and will only construct polylines if lines are available.

        Example:
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

        Parameters
        ----------
        verts : bool
            if True, break input vertex cells into individual vertex cells
            (one point per cell). If False, the input vertex cells will be ignored.

        lines : bool
            if True, break input polylines into line segments.
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
            vedo.logger.debug("input in triangulate() seems to be void")
            return self

    def compute_cell_area(self, name="Area"):
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

    def compute_cell_vertex_count(self, name="VertexCount"):
        """Add to this mesh a cell data array containing the nr of vertices
        that a polygonal face has."""
        csf = vtk.vtkCellSizeFilter()
        csf.SetInputData(self.polydata(False))
        csf.SetComputeArea(False)
        csf.SetComputeVolume(False)
        csf.SetComputeLength(False)
        csf.SetComputeVertexCount(True)
        csf.SetVertexCountArrayName(name)
        csf.Update()
        return self._update(csf.GetOutput())

    def compute_arc_length(self, mesh):
        """Given a mesh, add the length of the arc intersecting each point of the line."""
        arcl = vtk.vtkAppendArcLength()
        arcl.SetInputData(mesh.polydata())
        arcl.Update()
        return self._update(arcl.GetOutput())

    def compute_quality(self, measure=6):
        """
        Calculate functions of quality for the elements of a triangular mesh.
        This method adds to the mesh a cell array named "Quality".
        See class [vtkMeshQuality](https://vtk.org/doc/nightly/html/classvtkMeshQuality.html)
        for explanation.

        Parameters
        ----------
        measure : int
            type of estimator

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

        .. hint:: meshquality.py
            .. image:: https://vedo.embl.es/images/advanced/meshquality.png
        """
        qf = vtk.vtkMeshQuality()
        qf.SetInputData(self.polydata(False))
        qf.SetTriangleQualityMeasure(measure)
        qf.SaveCellQualityOn()
        qf.Update()
        pd = qf.GetOutput()
        self._update(pd)
        return self

    @deprecated(reason=vedo.colors.red + "Please use compute_curvature()" + vedo.colors.reset)
    def addCurvatureScalars(self, method=0):
        """Deprecated. Please use compute_curvature()"""
        return self.compute_curvature(method)

    def compute_curvature(self, method=0):
        """
        Add scalars to ``Mesh`` that contains the curvature calculated in three different ways.

        Use ``method`` as: 0-gaussian, 1-mean, 2-max, 3-min curvature.

        Example:
            .. code-block:: python

                from vedo import Torus
                Torus().addCurvatureScalars().addScalarBar().show(axes=1)

            .. image:: https://user-images.githubusercontent.com/32848391/51934810-c2e88c00-2404-11e9-8e7e-ca0b7984bbb7.png
        """
        curve = vtk.vtkCurvatures()
        curve.SetInputData(self._data)
        curve.SetCurvatureType(method)
        curve.Update()
        self._update(curve.GetOutput())
        self._mapper.ScalarVisibilityOn()
        return self

    def compute_connectivity(self):
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

    def compute_elevation(self, low=(0, 0, 0), high=(0, 0, 1), vrange=(0, 1)):
        """
        Add to ``Mesh`` a scalar array that contains distance along a specified direction.

        Parameters
        ----------
        low : list
            one end of the line (small scalar values).

        high : list
            other end of the line (large scalar values).

        vrange : list
            set the range of the scalar.

        Example:
            .. code-block:: python

                from vedo import Sphere
                s = Sphere().addElevationScalars(lowPoint=(0,0,0), highPoint=(1,1,1))
                s.addScalarBar().show(axes=1)

            .. image:: https://user-images.githubusercontent.com/32848391/68478872-3986a580-0231-11ea-8245-b68a683aa295.png
        """
        ef = vtk.vtkElevationFilter()
        ef.SetInputData(self.polydata())
        ef.SetLowPoint(low)
        ef.SetHighPoint(high)
        ef.SetScalarRange(vrange)
        ef.Update()
        self._update(ef.GetOutput())
        self._mapper.ScalarVisibilityOn()
        return self

    @deprecated(reason=vedo.colors.red + "Please use add_shadow()" + vedo.colors.reset)
    def addShadow(self, *a, **b):
        """Please use add_shadow()"""
        return self.add_shadow(*a, **b)

    def add_shadow(
        self,
        plane,
        point,
        direction=None,
        c=(0.6, 0.6, 0.6),
        alpha=1,
        culling=0,
    ):
        """
        Generate a shadow out of an ``Mesh`` on one of the three Cartesian planes.
        The output is a new ``Mesh`` representing the shadow.
        This new mesh is accessible through `mesh.shadow`.
        By default the shadow mesh is placed on the bottom wall of the bounding box.

        See also ``pointcloud.projectOnPlane``.

        Parameters
        ----------
        plane : str, Plane
            if plane is `str`, plane can be one of ['x', 'y', 'z'],
            represents x-plane, y-plane and z-plane, respectively.
            Otherwise, plane should be an instance of `vedo.shapes.Plane`.

        point : float, array
            if plane is `str`, point should be a float represents the intercept.
            Otherwise, point is the camera point of perspective projection

        direction : list
            direction of oblique projection

        culling : int
            choose between front [1] or backface [-1] culling or None.

        .. hint:: shadow.py, airplane1.py, airplane2.py
            .. image:: https://vedo.embl.es/images/simulations/57341963-b8910900-713c-11e9-898a-84b6d3712bce.gif
        """
        shad = self.clone()
        shad.name = "Shadow"
        pts = shad.points()
        if "x" == plane:
            # shad = shad.projectOnPlane('x')
            # instead do it manually so in case of alpha<1 we dont see glitches due to coplanar points
            # we leave a small tolerance of 0.1% in thickness
            x0, x1 = self.xbounds()
            pts[:, 0] = (pts[:, 0] - (x0 + x1) / 2) / 1000 + self.GetOrigin()[0]
            shad.points(pts)
            shad.x(point)
        elif "y" == plane:
            # shad = shad.projectOnPlane('y')
            x0, x1 = self.ybounds()
            pts[:, 1] = (pts[:, 1] - (x0 + x1) / 2) / 1000 + self.GetOrigin()[1]
            shad.points(pts)
            shad.y(point)
        elif "z" == plane:
            # shad = shad.projectOnPlane('z')
            x0, x1 = self.zbounds()
            pts[:, 2] = (pts[:, 2] - (x0 + x1) / 2) / 1000 + self.GetOrigin()[2]
            shad.points(pts)
            shad.z(point)
        else:
            shad = shad.projectOnPlane(plane, point, direction)

        shad.c(c).alpha(alpha).flat()

        if culling in (1, True):
            shad.frontFaceCulling()
        elif culling == -1:
            shad.backFaceCulling()

        shad.GetProperty().LightingOff()
        shad.SetPickable(False)
        shad.SetUseBounds(True)

        if shad not in self.shadows:
            self.shadows.append(shad)
            shad.info = dict(plane=plane, point=point, direction=direction)
        return self

    def subdivide(self, n=1, method=0, mel=None):
        """
        Increase the number of vertices of a surface mesh.

        Parameters
        ----------
        n : int
            number of subdivisions.

        method : int
            Loop(0), Linear(1), Adaptive(2), Butterfly(3)

        mel : float
            Maximum Edge Length (for Adaptive method only).
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
                mel = self.diagonal_size() / np.sqrt(self._data.GetNumberOfPoints()) / n
            sdf.SetMaximumEdgeLength(mel)
        elif method == 3:
            sdf = vtk.vtkButterflySubdivisionFilter()
        else:
            vedo.logger.error(f"in subdivide() unknown method {method}")
            raise RuntimeError()

        if method != 2:
            sdf.SetNumberOfSubdivisions(n)
        sdf.SetInputData(originalMesh)
        sdf.Update()
        return self._update(sdf.GetOutput())

    def decimate(self, fraction=0.5, n=None, method="quadric", boundaries=False):
        """
        Downsample the number of vertices in a mesh to `fraction`.

        Parameters
        ----------
        fraction : float
            the desired target of reduction.

        n : int
            the desired number of final points (`fraction` is recalculated based on it).

        method : str
            can be either 'quadric' or 'pro'. In the first case triagulation
            will look like more regular, irrespective of the mesh origianl curvature.
            In the second case triangles are more irregular but mesh is more precise on more
            curved regions.

        boundaries : bool
            in "pro" mode decide whether to leave boundaries untouched or not.

        .. note:: Setting ``fraction=0.1`` leaves 10% of the original nr of vertices.

        .. hint:: skeletonize.py
        """
        poly = self._data
        if n:  # N = desired number of points
            npt = poly.GetNumberOfPoints()
            fraction = n / npt
            if fraction >= 1:
                return self

        if "quad" in method:
            decimate = vtk.vtkQuadricDecimation()
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

    def collapse_edges(self, distance, iterations=1):
        """Collapse mesh edges so that are all above distance."""
        self.clean()
        x0, x1, y0, y1, z0, z1 = self.GetBounds()
        fs = min(x1 - x0, y1 - y0, z1 - z0) / 10
        d2 = distance * distance
        if distance > fs:
            vedo.logger.error(
                f"distance parameter is too large, should be < {fs}, skip!"
            )
            return self
        for _ in range(iterations):
            medges = self.edges()
            pts = self.points()
            newpts = np.array(pts)
            moved = []
            for e in medges:
                if len(e) == 2:
                    id0, id1 = e
                    p0, p1 = pts[id0], pts[id1]
                    d = mag2(p1 - p0)
                    if d < d2 and id0 not in moved and id1 not in moved:
                        p = (p0 + p1) / 2
                        newpts[id0] = p
                        newpts[id1] = p
                        moved += [id0, id1]

            self.points(newpts)
            self.clean()
        self.compute_normals()  # .flat()
        return self


    def smooth(
        self, niter=15, pass_band=0.1, edge_angle=15, feature_angle=60, boundary=False
    ):
        """
        Adjust mesh point positions using the `Windowed Sinc` function interpolation kernel.

        Parameters
        ----------
        niter : int
            number of iterations.

        pass_band : float
            set the pass_band value for the windowed sinc filter.

        edge_angle : float
            edge angle to control smoothing along edges (either interior or boundary).

        feature_angle : float
            specifies the feature angle for sharp edge identification.

        boundary : bool
            specify if boundary should also be smoothed or kept unmodified

        .. hint:: mesh_smoother1.py
            .. image:: https://vedo.embl.es/images/advanced/mesh_smoother2.png
        """
        poly = self._data
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        smoothFilter = vtk.vtkWindowedSincPolyDataFilter()
        smoothFilter.SetInputData(cl.GetOutput())
        smoothFilter.SetNumberOfIterations(niter)
        smoothFilter.SetEdgeAngle(edge_angle)
        smoothFilter.SetFeatureAngle(feature_angle)
        smoothFilter.SetPassBand(pass_band)
        smoothFilter.NormalizeCoordinatesOn()
        smoothFilter.NonManifoldSmoothingOn()
        smoothFilter.FeatureEdgeSmoothingOn()
        smoothFilter.SetBoundarySmoothing(boundary)
        smoothFilter.Update()
        return self._update(smoothFilter.GetOutput())

    @deprecated(reason=vedo.colors.red + "Please use fill_holes()" + vedo.colors.reset)
    def fillHoles(self, size=None):
        """Deprecated. Please use fill_holes()"""
        return self.fill_holes(size)

    def fill_holes(self, size=None):
        """
        Identifies and fills holes in input mesh.
        Holes are identified by locating boundary edges, linking them together into loops,
        and then triangulating the resulting loops.

        Parameters
        ----------
        size : float, optional
            Approximate limit to the size of the hole that can be filled. The default is None.

        .. hint:: fillholes.py
        """
        fh = vtk.vtkFillHolesFilter()
        if not size:
            mb = self.diagonal_size()
            size = mb / 10
        fh.SetHoleSize(size)
        fh.SetInputData(self._data)
        fh.Update()
        return self._update(fh.GetOutput())

    def is_inside(self, point, tol=1e-05):
        """Return True if point is inside a polydata closed surface."""
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

    def inside_points(self, pts, invert=False, tol=1e-05, return_ids=False):
        """
        Return the point cloud that is inside mesh surface as a new Points object.

        If return_ids is True a list of IDs is returned and in addition input points
        are marked by a pointdata array named "IsInside".
        E.g.: print(pts.pointdata["IsInside"])

        .. hint:: examples/basic/pca_ellipsoid.py
            .. image:: https://vedo.embl.es/images/basic/pca.png
        """
        if isinstance(pts, Points):
            pointsPolydata = pts.polydata()
            ptsa = pts.points()
        else:
            ptsa = np.asarray(pts, dtype=float)
            vpoints = vtk.vtkPoints()
            vpoints.SetData(numpy2vtk(ptsa, dtype=float))
            pointsPolydata = vtk.vtkPolyData()
            pointsPolydata.SetPoints(vpoints)

        sep = vtk.vtkSelectEnclosedPoints()
        # sep = vtk.vtkExtractEnclosedPoints()
        sep.SetTolerance(tol)
        sep.SetInputData(pointsPolydata)
        sep.SetSurfaceData(self.polydata())
        sep.SetInsideOut(invert)
        sep.Update()

        varr = sep.GetOutput().GetPointData().GetArray("SelectedPoints")
        mask = vtk2numpy(varr).astype(np.bool)
        ids = np.array(range(len(ptsa)), dtype=int)[mask]

        if isinstance(pts, Points):
            varr.SetName("IsInside")
            pts.inputdata().GetPointData().AddArray(varr)

        if return_ids:
            return ids
        else:
            pcl = Points(ptsa[ids])
            pcl.name = "InsidePoints"
            return pcl

    def boundaries(
        self,
        boundary_edges=True,
        manifold_edges=False,
        non_manifold_edges=False,
        feature_angle=None,
        return_point_ids=False,
        return_cell_ids=False,
    ):
        """
        Return the boundary lines of an input mesh.

        Parameters
        ----------
        boundary_edges : bool
            Turn on/off the extraction of boundary edges.

        manifold_edges : bool
            Turn on/off the extraction of manifold edges.

        non_manifold_edges : bool
            Turn on/off the extraction of non-manifold edges.

        feature_angle : bool
            Specify the min angle btw 2 faces for extracting edges.

        return_point_ids : bool
            return a numpy array of point indices

        return_cell_ids : bool
            return a numpy array of cell indices

        .. hint:: boundaries.py
            .. image:: https://vedo.embl.es/images/basic/boundaries.png
        """
        fe = vtk.vtkFeatureEdges()
        fe.SetBoundaryEdges(boundary_edges)
        fe.SetNonManifoldEdges(non_manifold_edges)
        fe.SetManifoldEdges(manifold_edges)
        # fe.SetPassLines(True) # vtk9.2
        fe.ColoringOff()
        fe.SetFeatureEdges(False)
        if feature_angle is not None:
            fe.SetFeatureEdges(True)
            fe.SetFeatureAngle(feature_angle)

        if return_point_ids or return_cell_ids:
            idf = vtk.vtkIdFilter()
            idf.SetInputData(self.polydata())
            idf.SetPointIdsArrayName("BoundaryIds")
            idf.SetPointIds(True)
            idf.Update()

            fe.SetInputData(idf.GetOutput())
            fe.Update()

            vid = fe.GetOutput().GetPointData().GetArray("BoundaryIds")
            npid = vtk2numpy(vid).astype(int)

            if return_point_ids:
                return npid

            if return_cell_ids:
                inface = []
                for i, face in enumerate(self.faces()):
                    isin = np.all([vtx in npid for vtx in face])
                    if isin:
                        inface.append(i)
                return np.array(inface).astype(int)

        else:

            fe.SetInputData(self.polydata())
            fe.Update()
            return Mesh(fe.GetOutput(), c="p").lw(5).lighting("off")


    def imprint(self, loopline, tol=0.01):
        """
        Imprint the contact surface of one object onto another surface.

        Parameters
        ----------
        loopline : vedo.shapes.Line
            a Line object to be imprinted onto the mesh.

        tol : float, optional
            projection tolerance which controls how close the imprint surface must be to the target.

        Example:
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

    def connected_vertices(self, index):
        """Find all vertices connected to an input vertex specified by its index.

        .. hint:: connVtx.py
            .. image:: https://vedo.embl.es/images/basic/connVtx.png
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

        return idxs

    def connected_cells(self, index, return_ids=False):
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
        if return_ids:
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

    @deprecated(reason=vedo.colors.red + "Please use intersect_with_line()" + vedo.colors.reset)
    def intersectWithLine(self, p0, p1=None, returnIds=False, tol=0):
        "Deprecated. Please use intersect_with_line()"
        return self.intersect_with_line(p0, p1, returnIds, tol)

    def intersect_with_line(self, p0, p1=None, return_ids=False, tol=0):
        """
        Return the list of points intersecting the mesh
        along the segment defined by two points `p0` and `p1`.

        Use ``return_ids`` to return the cell ids instead of point coords

        Example:
            .. code-block:: python

                from vedo import *
                s = Spring()
                pts = s.intersect_with_line([0,0,0], [1,0.1,0])
                ln = Line([0,0,0], [1,0.1,0], c='blue')
                ps = Points(pts, r=10, c='r')
                show(s, ln, ps, bg='white')

            .. image:: https://user-images.githubusercontent.com/32848391/55967065-eee08300-5c79-11e9-8933-265e1bab9f7e.png
        """
        if isinstance(p0, Points):
            p0, p1 = p0.points()

        if not self.line_locator:
            self.line_locator = vtk.vtkOBBTree()
            self.line_locator.SetDataSet(self.polydata())
            if not tol:
                tol = mag(np.asarray(p1) - np.asarray(p0)) / 10000
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

        if return_ids:
            pts_ids = []
            for i in range(idlist.GetNumberOfIds()):
                cid = idlist.GetId(i)
                pts_ids.append([pts[i], cid])
            return np.array(pts_ids)
        else:
            return pts

    def silhouette(self, direction=None, border_edges=True, feature_angle=False):
        """
        Return a new line ``Mesh`` which corresponds to the outer `silhouette`
        of the input as seen along a specified `direction`, this can also be
        a ``vtkCamera`` object.

        Parameters
        ----------
        direction : list
            viewpoint direction vector.
            If *None* this is guessed by looking at the minimum
            of the sides of the bounding box.

        border_edges : bool
            enable or disable generation of border edges

        feature_angle : float
            minimal angle for sharp edges detection.
            If set to `False` the functionality is disabled.

        .. hint:: silhouette.py
            .. image:: https://vedo.embl.es/images/basic/silhouette1.png
        """
        sil = vtk.vtkPolyDataSilhouette()
        sil.SetInputData(self.polydata())
        sil.SetBorderEdges(border_edges)
        if feature_angle is False:
            sil.SetEnableFeatureAngle(0)
        else:
            sil.SetEnableFeatureAngle(1)
            sil.SetFeatureAngle(feature_angle)

        if direction is None and vedo.plotter_instance and vedo.plotter_instance.camera:
            sil.SetCamera(vedo.plotter_instance.camera)
            m = Mesh()
            m.mapper().SetInputConnection(sil.GetOutputPort())

        elif isinstance(direction, vtk.vtkCamera):
            sil.SetCamera(direction)
            m = Mesh()
            m.mapper().SetInputConnection(sil.GetOutputPort())

        elif direction == "2d":
            sil.SetVector(3.4, 4.5, 5.6)  # random
            sil.SetDirectionToSpecifiedVector()
            sil.Update()
            m = Mesh(sil.GetOutput())

        elif is_sequence(direction):
            sil.SetVector(direction)
            sil.SetDirectionToSpecifiedVector()
            sil.Update()
            m = Mesh(sil.GetOutput())
        else:
            vedo.logger.error(f"in silhouette() unknown direction type {type(direction)}")
            vedo.logger.error("first render the scene with show() or specify camera/direction")
            return self

        m.lw(2).c((0, 0, 0)).lighting("off")
        m.mapper().SetResolveCoincidentTopologyToPolygonOffset()
        return m

    @deprecated(reason=vedo.colors.red + "Please use follow_camera()" + vedo.colors.reset)
    def followCamera(self, cam=None):
        "Deprecated. Please use follow_camera()"
        return self.follow_camera(cam)

    def follow_camera(self, cam=None):
        """
        Mesh object will follow camera movements and stay locked to it.
        Use ``mesh.followCamera(False)`` to disable it.

        Set ``cam`` to None and the text will auto-orient itself to the active camera.
        A ``vtkCamera`` object can also be passed.
        """
        if cam is False:
            self.SetCamera(None)
            return self
        if isinstance(cam, vtk.vtkCamera):
            self.SetCamera(cam)
        else:
            plt = vedo.plotter_instance
            if plt and plt.camera:
                self.SetCamera(plt.camera)
            else:
                # postpone to show() call
                self._set2actcam = True
        return self

    def isobands(self, n=10, vmin=None, vmax=None):
        """
        Return a new ``Mesh`` representing the isobands of the active scalars.
        This is a new mesh where the scalar is now associated to cell faces and
        used to colorize the mesh.

        Parameters
        ----------
        n : int
            number of isobands in the range

        vmin : float
            minimum of the range

        vmax : float
            maximum of the range

        .. hint:: isolines.py
            .. image:: https://vedo.embl.es/images/pyplot/isolines.png
        """
        r0, r1 = self._data.GetScalarRange()
        if vmin is None:
            vmin = r0
        if vmax is None:
            vmax = r1

        # --------------------------------
        bands = []
        dx = (vmax - vmin) / float(n)
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
            labels.append("{:4.2f}".format(b[1]))
        values = vtk.vtkVariantArray()
        for la in labels:
            values.InsertNextValue(vtk.vtkVariant(la))
        for i in range(values.GetNumberOfTuples()):
            lut.SetAnnotation(i, values.GetValue(i).ToString())

        bcf = vtk.vtkBandedPolyDataContourFilter()
        bcf.SetInputData(self.polydata())
        # Use either the minimum or maximum value for each band.
        for i, band in enumerate(bands):
            bcf.SetValue(i, band[2])
        # We will use an indexed lookup table.
        bcf.SetScalarModeToIndex()
        bcf.GenerateContourEdgesOff()
        bcf.Update()
        bcf.GetOutput().GetCellData().GetScalars().SetName("IsoBands")
        m1 = Mesh(bcf.GetOutput()).compute_normals(cells=True)
        m1.mapper().SetLookupTable(lut)
        return m1

    def isolines(self, n=10, vmin=None, vmax=None):
        """
        Return a new ``Mesh`` representing the isolines of the active scalars.

        Parameters
        ----------
        n : int
            number of isolines in the range

        vmin : float
            minimum of the range

        vmax : float
            maximum of the range

        .. hint:: isolines.py
            .. image:: https://vedo.embl.es/images/pyplot/isolines.png
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
        msh = Mesh(cl.GetOutput(), c="k").lighting("off")
        msh.mapper().SetResolveCoincidentTopologyToPolygonOffset()
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

        .. hint:: extrude.py
            .. image:: https://vedo.embl.es/images/basic/extrude.png
        """
        if is_sequence(zshift):
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
            rf.SetInputData(self.polydata(False))  # must not be transformed
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
            return m.compute_normals(cells=False).flat().lighting("default")

    def split(self, maxdepth=1000, flag=False):
        """
        Split a mesh by connectivity and order the pieces by increasing area.

        Parameters
        ----------
        maxdepth : int
            only consider this maximum number of mesh parts.

        flag : bool
            if set to True return the same single object,
            but add a "RegionId" array to flag the mesh subparts

        .. hint:: splitmesh.py
            .. image:: https://vedo.embl.es/images/advanced/splitmesh.png
        """
        pd = self.polydata(False)
        cf = vtk.vtkConnectivityFilter()
        cf.SetInputData(pd)
        cf.SetExtractionModeToAllRegions()
        cf.SetColorRegions(True)
        cf.Update()

        if flag:
            return self._update(cf.GetOutput())

        a = Mesh(cf.GetOutput())
        alist = []

        for t in range(max(a.pointdata["RegionId"]) + 1):
            if t == maxdepth:
                break
            suba = a.clone().threshold("RegionId", t - 0.1, t + 0.1)
            area = suba.area()
            # print('splitByConnectivity  piece:', t, ' area:', area, ' N:',suba.npoints)
            alist.append([suba, area])

        alist.sort(key=lambda x: x[1])
        alist.reverse()
        blist = []
        for i, l in enumerate(alist):
            l[0].color(i + 1).phong()
            l[0].mapper().ScalarVisibilityOff()
            blist.append(l[0])
        return blist

    def extract_largest_region(self):
        """
        Extract the largest connected part of a mesh and discard all the smaller pieces.

        .. hint:: largestregion.py
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
        m.mapper().SetScalarVisibility(vis)
        return m

    def boolean(self, operation, mesh2):
        """Volumetric union, intersection and subtraction of surfaces.

        Use ``operation`` for the allowed operations 'plus', 'intersect', 'minus'.

        .. hint:: boolean.py
            .. image:: https://vedo.embl.es/images/basic/boolean.png
        """
        bf = vtk.vtkBooleanOperationPolyDataFilter()
        poly1 = self.compute_normals().polydata()
        poly2 = mesh2.compute_normals().polydata()
        if operation.lower() == "plus" or operation.lower() == "+":
            bf.SetOperationToUnion()
        elif operation.lower() == "intersect":
            bf.SetOperationToIntersection()
        elif operation.lower() == "minus" or operation.lower() == "-":
            bf.SetOperationToDifference()
        # bf.ReorientDifferenceCellsOn()
        bf.SetInputData(0, poly1)
        bf.SetInputData(1, poly2)
        bf.Update()
        mesh = Mesh(bf.GetOutput(), c=None)
        mesh.flat()
        mesh.name = self.name + operation + mesh2.name
        return mesh

    def intersect_with(self, mesh2, tol=1e-06):
        """
        Intersect this Mesh with the input surface to return a set of lines.

        .. hint:: surfIntersect.py
            .. image:: https://vedo.embl.es/images/basic/surfIntersect.png
        """
        bf = vtk.vtkIntersectionPolyDataFilter()
        bf.SetGlobalWarningDisplay(0)
        if isinstance(self, Mesh):
            poly1 = self.polydata()
        else:
            poly1 = self.GetMapper().GetInput()
        if isinstance(mesh2, Mesh):
            poly2 = mesh2.polydata()
        else:
            poly2 = mesh2.GetMapper().GetInput()
        bf.SetTolerance(tol)
        bf.SetInputData(0, poly1)
        bf.SetInputData(1, poly2)
        bf.Update()
        mesh = Mesh(bf.GetOutput(), "k", 1).lighting("off")
        mesh.GetProperty().SetLineWidth(3)
        mesh.name = "surfaceIntersection"
        return mesh

    def geodesic(self, start, end):
        """
        Dijkstra algorithm to compute the geodesic line.
        Takes as input a polygonal mesh and performs a single source shortest path calculation.

        Parameters
        ----------
        start : int, list
            start vertex index or close point `[x,y,z]`

        end :  int, list
            end vertex index or close point `[x,y,z]`

        .. hint:: geodesic.py
            .. image:: https://vedo.embl.es/images/advanced/geodesic.png
        """
        if is_sequence(start):
            cc = self.points()
            pa = Points(cc)
            start = pa.closest_point(start, return_point_id=True)
            end = pa.closest_point(end, return_point_id=True)

        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(self.polydata())
        dijkstra.SetStartVertex(end)  # inverted in vtk
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

        dmesh = Mesh(poly, c="k")
        prop = vtk.vtkProperty()
        prop.DeepCopy(self.property)
        prop.SetLineWidth(3)
        prop.SetOpacity(1)
        dmesh.SetProperty(prop)
        dmesh.property = prop
        dmesh.name = "GeodesicLine"
        return dmesh

    #####################################################################
    ### Stuff returning a Volume
    ###
    def binarize(self, spacing=(1, 1, 1), invert=False):
        """
        Convert a ``Mesh`` into a ``Volume``
        where the foreground (exterior) voxels value is 255 and the background
        (interior) voxels value is 0.

        .. hint:: mesh2volume.py
            .. image:: https://vedo.embl.es/images/volumetric/mesh2volume.png
        """
        # https://vtk.org/Wiki/VTK/Examples/Cxx/PolyData/PolyDataToImageData
        pd = self.polydata()

        whiteImage = vtk.vtkImageData()
        bounds = pd.GetBounds()

        whiteImage.SetSpacing(spacing)

        # compute dimensions
        dim = [0, 0, 0]
        for i in [0, 1, 2]:
            dim[i] = int(np.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i]))
        whiteImage.SetDimensions(dim)
        whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)

        origin = [0, 0, 0]
        origin[0] = bounds[0] + spacing[0] / 2
        origin[1] = bounds[2] + spacing[1] / 2
        origin[2] = bounds[4] + spacing[2] / 2
        whiteImage.SetOrigin(origin)
        whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # fill the image with foreground voxels:
        if invert:
            inval = 0
        else:
            inval = 255
        count = whiteImage.GetNumberOfPoints()
        for i in range(count):
            whiteImage.GetPointData().GetScalars().SetTuple1(i, inval)

        # polygonal data --> image stencil:
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(pd)
        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        # cut the corresponding white image and set the background:
        if invert:
            outval = 255
        else:
            outval = 0
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(whiteImage)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.SetReverseStencil(invert)
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()
        return vedo.Volume(imgstenc.GetOutput())


    @deprecated(reason=vedo.colors.red + "Please use signed_distance()" + vedo.colors.reset)
    def signedDistance(self, **k):
        "Deprecated. Please use signed_distance()"
        return self.signed_distance(**k)

    def signed_distance(self, bounds=None, dims=(20, 20, 20), invert=False, maxradius=None):
        """
        Compute the ``Volume`` object whose voxels contains the signed distance from
        the mesh.

        Parameters
        ----------
        bounds : list
            bounds of the output volume.

        dims : list
            dimensions (nr. of voxels) of the output volume.

        invert : bool
            flip the sign.

        .. hint:: examples/volumetric/volumeFromMesh.py
        """
        if maxradius is not None:
            vedo.logger.warning(
                "in signedDistance(maxradius=...) is ignored. (Only valid for pointclouds)."
            )
        if bounds is None:
            bounds = self.GetBounds()
        sx = (bounds[1] - bounds[0]) / dims[0]
        sy = (bounds[3] - bounds[2]) / dims[1]
        sz = (bounds[5] - bounds[4]) / dims[2]

        img = vtk.vtkImageData()
        img.SetDimensions(dims)
        img.SetSpacing(sx, sy, sz)
        img.SetOrigin(bounds[0], bounds[2], bounds[4])
        img.AllocateScalars(vtk.VTK_FLOAT, 1)

        imp = vtk.vtkImplicitPolyDataDistance()
        imp.SetInput(self.polydata())
        b2 = bounds[2]
        b4 = bounds[4]
        d0, d1, d2 = dims

        for i in range(d0):
            x = i * sx + bounds[0]
            for j in range(d1):
                y = j * sy + b2
                for k in range(d2):
                    v = imp.EvaluateFunction((x, y, k * sz + b4))
                    if invert:
                        v = -v
                    img.SetScalarComponentFromFloat(i, j, k, 0, v)

        vol = vedo.Volume(img)
        vol.name = "SignedVolume"
        return vol

    def tetralize(
        self,
        side=0.02,
        nmax=300_000,
        gap=None,
        subsample=False,
        uniform=True,
        seed=0,
        debug=False,
    ):
        """
        Tetralize a closed polygonal mesh. Return a `TetMesh`.

        Parameters
        ----------
        side : float
            desired side of the single tetras as fraction of the bounding box diagonal.
            Typical values are in the range (0.01 - 0.03)

        nmax : int
            maximum random numbers to be sampled in the bounding box

        gap : float
            keep this minimum distance from the surface,
            if None an automatic choice is made.

        subsample : bool
            subsample input surface, the geometry might be affected
            (the number of original faces reduceed), but higher tet quality might be obtained.

        uniform : bool
            generate tets more uniformly packed in the interior of the mesh

        seed : int
            random number generator seed

        debug : bool
            show an intermediate plot with sampled points

        .. hint:: examples/volumetric/tetralize_surface.py
        """
        surf = self.clone().clean().compute_normals()
        d = surf.diagonal_size()
        if gap is None:
            gap = side * d * np.sqrt(2 / 3)
        n = int(min((1 / side) ** 3, nmax))

        # fill the space w/ points
        x0, x1, y0, y1, z0, z1 = surf.bounds()

        if uniform:
            pts = vedo.utils.pack_spheres([x0, x1, y0, y1, z0, z1], side * d * 1.42)
            pts += np.random.randn(len(pts), 3) * side * d * 1.42 / 100  # some small jitter
        else:
            disp = np.array([x0 + x1, y0 + y1, z0 + z1]) / 2
            np.random.seed(seed)
            pts = (np.random.rand(n, 3) - 0.5) * np.array(
                [x1 - x0, y1 - y0, z1 - z0]
            ) + disp

        normals = surf.celldata["Normals"]
        cc = surf.cellCenters()
        subpts = cc - normals * gap * 1.05
        pts = pts.tolist() + subpts.tolist()

        if debug:
            print(".. tetralize(): subsampling and cleaning")

        fillpts = surf.inside_points(pts)
        fillpts.subsample(side)

        if gap:
            fillpts.distance_to(surf)
            fillpts.threshold("Distance", above=gap)

        if subsample:
            surf.subsample(side)

        tmesh = vedo.tetmesh.delaunay3d(vedo.merge(fillpts, surf))
        tcenters = tmesh.cellCenters()

        ids = surf.inside_points(tcenters, return_ids=True)
        ins = np.zeros(tmesh.NCells())
        ins[ids] = 1

        if debug:
            # vedo.pyplot.histogram(fillpts.pointdata["Distance"], xtitle=f"gap={gap}").show().close()
            edges = self.edges()
            points = self.points()
            elen = mag(points[edges][:, 0, :] - points[edges][:, 1, :])
            histo = vedo.pyplot.histogram(elen, xtitle="edge length", xlim=(0, 3 * side * d))
            print(".. edges min, max", elen.min(), elen.max())
            fillpts.cmap("bone")
            vedo.show([
                    [   f"This is a debug plot.\n\nGenerated points: {n}\ngap: {gap}",
                        surf.wireframe().alpha(0.2),
                        vedo.addons.Axes(surf),
                        fillpts,
                        Points(subpts).c("r4").ps(3),
                    ],
                    [   histo,
                        f"Edges mean length: {np.mean(elen)}\n\nPress q to continue",
                    ],
                ],
                N=2,
                sharecam=False,
                new=True,
            ).close()
            print(".. thresholding")

        tmesh.celldata["inside"] = ins.astype(np.uint8)
        tmesh.threshold("inside", above=0.9)
        tmesh.celldata.remove("inside")

        if debug:
            print(f".. tetralize() completed, ntets = {tmesh.NCells()}")
        return tmesh
