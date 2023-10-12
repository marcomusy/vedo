#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import vedo
from vedo.colors import color_map
from vedo.colors import get_color
from vedo.pointcloud import Points
from vedo.utils import buildPolyData, is_sequence, mag, mag2, precision
from vedo.utils import numpy2vtk, vtk2numpy, OperationNode

__docformat__ = "google"

__doc__ = """
Submodule to work with polygonal meshes

![](https://vedo.embl.es/images/advanced/mesh_smoother2.png)
"""

__all__ = ["Mesh"]

class MeshVisual:
    """Class to manage the visual aspects of a ``Maesh`` object."""

    def follow_camera(self, camera=None, origin=None):
        """
        Return an object that will follow camera movements and stay locked to it.
        Use `mesh.follow_camera(False)` to disable it.

        A `vtkCamera` object can also be passed.
        """
        if camera is False:
            try:
                self.SetCamera(None)
                return self
            except AttributeError:
                return self

        factor = vtk.vtkFollower()
        factor.SetMapper(self.mapper)
        factor.SetProperty(self.property)
        factor.SetBackfaceProperty(self.actor.GetBackfaceProperty())
        factor.SetTexture(self.actor.GetTexture())
        factor.SetScale(self.actor.GetScale())
        factor.SetOrientation(self.actor.GetOrientation())
        factor.SetPosition(self.actor.GetPosition())
        factor.SetUseBounds(self.actor.GetUseBounds())

        factor.SetOrigin(self.actor.GetOrigin())

        factor.PickableOff()

        if isinstance(camera, vtk.vtkCamera):
            factor.SetCamera(camera)
        else:
            plt = vedo.plotter_instance
            if plt and plt.renderer and plt.renderer.GetActiveCamera():
                factor.SetCamera(plt.renderer.GetActiveCamera())
        
        if origin is not None:
            factor.SetOrigin(origin)

        self.actor = None
        factor.data = self
        self.actor = factor
        return self


    def wireframe(self, value=True):
        """Set mesh's representation as wireframe or solid surface."""
        if value:
            self.property.SetRepresentationToWireframe()
        else:
            self.property.SetRepresentationToSurface()
        return self

    def flat(self):
        """Set surface interpolation to flat.

        <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Phong_components_version_4.png" width="700">
        """
        self.property.SetInterpolationToFlat()
        return self

    def phong(self):
        """Set surface interpolation to "phong"."""
        self.property.SetInterpolationToPhong()
        return self

    def backface_culling(self, value=True):
        """Set culling of polygons based on orientation of normal with respect to camera."""
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

    def backcolor(self, bc=None):
        """
        Set/get mesh's backface color.
        """
        back_prop = self.actor.GetBackfaceProperty()

        if bc is None:
            if back_prop:
                return back_prop.GetDiffuseColor()
            return self

        if self.property.GetOpacity() < 1:
            return self

        if not back_prop:
            back_prop = vtk.vtkProperty()

        back_prop.SetDiffuseColor(get_color(bc))
        back_prop.SetOpacity(self.property.GetOpacity())
        self.actor.SetBackfaceProperty(back_prop)
        self.mapper.ScalarVisibilityOff()
        return self

    def bc(self, backcolor=False):
        """Shortcut for `mesh.backcolor()`."""
        return self.backcolor(backcolor)

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
        self.property.EdgeVisibilityOn()
        self.property.SetEdgeColor(get_color(lc))
        return self

    def lc(self, linecolor=None):
        """Set/get color of mesh edges. Same as `linecolor()`."""
        return self.linecolor(linecolor)


####################################################
class Mesh(MeshVisual, Points):
    """
    Build an instance of object `Mesh` derived from `vedo.PointCloud`.
    """

    def __init__(self, inputobj=None, c='gold', alpha=1):
        """
        Input can be a list of vertices and their connectivity (faces of the polygonal mesh),
        or directly a `vtkPolydata` object.
        For point clouds - e.i. no faces - just substitute the `faces` list with `None`.

        Example:
            `Mesh( [ [[x1,y1,z1],[x2,y2,z2], ...],  [[0,1,2], [1,2,3], ...] ] )`

        Arguments:
            c : (color)
                color in RGB format, hex, symbol or name
            alpha : (float)
                mesh opacity [0,1]

        Examples:
            - [buildmesh.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/buildmesh.py)
            (and many others!)

            ![](https://vedo.embl.es/images/basic/buildmesh.png)
        """
        super().__init__()

        if inputobj is None:
            pass

        elif isinstance(inputobj, vtk.vtkActor):
            self.dataset.DeepCopy(inputobj.GetMapper().GetInput())
            v = inputobj.GetMapper().GetScalarVisibility()
            self.mapper.SetScalarVisibility(v)
            pr = vtk.vtkProperty()
            pr.DeepCopy(inputobj.GetProperty())
            self.actor.SetProperty(pr)
            self.property = pr

        elif isinstance(inputobj, vtk.vtkPolyData):
            if inputobj.GetNumberOfCells() == 0:
                carr = vtk.vtkCellArray()
                for i in range(inputobj.GetNumberOfPoints()):
                    carr.InsertNextCell(1)
                    carr.InsertCellPoint(i)
                inputobj.SetVerts(carr)
            self.dataset.DeepCopy(inputobj)

        elif is_sequence(inputobj):
            ninp = len(inputobj)
            if ninp == 2:  # assume input is [vertices, faces]
                self.dataset = buildPolyData(inputobj[0], inputobj[1])
            else:          # assume input is [vertices]
                self.dataset = buildPolyData(inputobj, None)

        elif isinstance(inputobj, str):
            self.dataset = vedo.file_io.load(inputobj).dataset
            self.filename = inputobj

        elif isinstance(inputobj, (vtk.vtkStructuredGrid, vtk.vtkRectilinearGrid)):
            gf = vtk.vtkGeometryFilter()
            gf.SetInputData(inputobj)
            gf.Update()
            self.dataset = gf.GetOutput()

        elif "meshlab" in str(type(inputobj)):
            self.dataset = vedo.utils.meshlab2vedo(inputobj)

        elif "trimesh" in str(type(inputobj)):
            self.dataset = vedo.utils.trimesh2vedo(inputobj)

        elif "meshio" in str(type(inputobj)):
            # self.dataset = vedo.utils.meshio2vedo(inputobj) ##TODO
            if len(inputobj.cells) > 0:
                mcells = []
                for cellblock in inputobj.cells:
                    if cellblock.type in ("triangle", "quad"):
                        mcells += cellblock.data.tolist()
                self.dataset = buildPolyData(inputobj.points, mcells)
            else:
                self.dataset = buildPolyData(inputobj.points, None)
            # add arrays:
            try:
                if len(inputobj.point_data) > 0:
                    for k in inputobj.point_data.keys():
                        vdata = numpy2vtk(inputobj.point_data[k])
                        vdata.SetName(str(k))
                        self.dataset.GetPointData().AddArray(vdata)
            except AssertionError:
                print("Could not add meshio point data, skip.")

        else:
            try:
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(inputobj)
                gf.Update()
                self.dataset = gf.GetOutput()
            except:
                vedo.logger.error(f"cannot build mesh from type {type(inputobj)}")
                raise RuntimeError()

        self.mapper.SetInputData(self.dataset)
        self.actor.SetMapper(self.mapper)

        self.property.SetInterpolationToPhong()
        self.property.SetColor(get_color(c))

        if alpha is not None:
            self.property.SetOpacity(alpha)

        self.mapper.SetInterpolateScalarsBeforeMapping(
            vedo.settings.interpolate_scalars_before_mapping
        )

        if vedo.settings.use_polygon_offset:
            self.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            pof = vedo.settings.polygon_offset_factor
            pou = vedo.settings.polygon_offset_units
            self.mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(pof, pou)

        n = self.dataset.GetNumberOfPoints()
        self.pipeline = OperationNode(self, comment=f"#pts {n}")
        self._texture = None

    def _repr_html_(self):
        """
        HTML representation of the Mesh object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.mesh.Mesh"
        help_url = "https://vedo.embl.es/docs/vedo/mesh.html"

        arr = self.thumbnail()
        im = Image.fromarray(arr)
        buffered = io.BytesIO()
        im.save(buffered, format="PNG", quality=100)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = "data:image/png;base64," + encoded
        image = f"<img src='{url}'></img>"

        bounds = "<br/>".join(
            [
                precision(min_x, 4) + " ... " + precision(max_x, 4)
                for min_x, max_x in zip(self.bounds()[::2], self.bounds()[1::2])
            ]
        )
        average_size = "{size:.3f}".format(size=self.average_size())

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

        allt = [
            "<table>",
            "<tr>",
            "<td>",
            image,
            "</td>",
            "<td style='text-align: center; vertical-align: center;'><br/>",
            help_text,
            "<table>",
            "<tr><td><b> bounds </b> <br/> (x/y/z) </td><td>" + str(bounds) + "</td></tr>",
            "<tr><td><b> center of mass </b></td><td>"
            + precision(self.center_of_mass(), 3)
            + "</td></tr>",
            "<tr><td><b> average size </b></td><td>" + str(average_size) + "</td></tr>",
            "<tr><td><b> nr. points&nbsp/&nbspfaces </b></td><td>"
            + str(self.npoints)
            + "&nbsp/&nbsp"
            + str(self.ncells)
            + "</td></tr>",
            pdata,
            cdata,
            "</table>",
            "</table>",
        ]
        return "\n".join(allt)

    def faces(self, ids=()):
        """
        DEPRECATED. Use property `mesh.faces` instead.

        Get cell polygonal connectivity ids as a python `list`.
        The output format is: `[[id0 ... idn], [id0 ... idm],  etc]`.

        If ids is set, return only the faces of the given cells.
        """
        arr1d = vtk2numpy(self.dataset.GetPolys().GetData())

        # Get cell connettivity ids as a 1D array. vtk format is:
        # [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        if len(arr1d) == 0:
            arr1d = vtk2numpy(self.dataset.GetStrips().GetData())

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
        if len(ids):
            return conn[ids]
        return conn  # cannot always make a numpy array of it!

    @property
    def cells(self):
        """
        Get cell polygonal connectivity ids as a python `list`.
        The output format is: `[[id0 ... idn], [id0 ... idm],  etc]`.

        If ids is set, return only the faces of the given cells.
        """
        arr1d = vtk2numpy(self.dataset.GetPolys().GetData())

        # Get cell connettivity ids as a 1D array. vtk format is:
        # [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        if len(arr1d) == 0:
            arr1d = vtk2numpy(self.dataset.GetStrips().GetData())

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

    @property
    def lines(self):
        """
        Get lines connectivity ids as a numpy array.
        Default format is `[[id0,id1], [id3,id4], ...]`

        Arguments:
            flat : (bool)
                return a 1D numpy array as e.g. [2, 10,20, 3, 10,11,12, 2, 70,80, ...]
        """
        # Get cell connettivity ids as a 1D array. The vtk format is:
        #    [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        arr1d = vtk2numpy(self.dataset.GetLines().GetData())
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

    @property
    def lines_as_flat_array(self):
        """
        Get lines connectivity ids as a numpy array.
        Format is e.g. [2,  10,20,  3, 10,11,12,  2, 70,80, ...]
        """
        return vtk2numpy(self.dataset.GetLines().GetData())

    @property
    def edges(self):
        """
        Return an array containing the edges connectivity.

        If ids is set, return only the edges of the given cells.
        """
        extractEdges = vtk.vtkExtractEdges()
        extractEdges.SetInputData(self.dataset)
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
        If tname is set to `None` texture is disabled.
        Input tname can also be an array or a `vtkTexture`.

        Arguments:
            tname : (numpy.array, str, Picture, vtkTexture, None)
                the input texture to be applied. Can be a numpy array, a path to an image file,
                a vedo Picture. The None value disables texture.
            tcoords : (numpy.array, str)
                this is the (u,v) texture coordinate array. Can also be a string of an existing array
                in the mesh.
            interpolate : (bool)
                turn on/off linear interpolation of the texture map when rendering.
            repeat : (bool)
                repeat of the texture when tcoords extend beyond the [0,1] range.
            edge_clamp : (bool)
                turn on/off the clamping of the texture map when
                the texture coords extend beyond the [0,1] range.
                Only used when repeat is False, and edge clamping is supported by the graphics card.
            scale : (bool)
                scale the texture image by this factor
            ushift : (bool)
                shift u-coordinates of texture by this amount
            vshift : (bool)
                shift v-coordinates of texture by this amount
            seam_threshold : (float)
                try to seal seams in texture by collapsing triangles
                (test values around 1.0, lower values = stronger collapse)

        Examples:
            - [texturecubes.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/texturecubes.py)

            ![](https://vedo.embl.es/images/basic/texturecubes.png)
        """
        pd = self.dataset
        out_img = None

        if tname is None:  # disable texture
            pd.GetPointData().SetTCoords(None)
            pd.GetPointData().Modified()
            return self  ######################################

        if isinstance(tname, vtk.vtkTexture):
            tu = tname

        elif isinstance(tname, vedo.Picture):
            tu = vtk.vtkTexture()
            out_img = tname

        elif is_sequence(tname):
            tu = vtk.vtkTexture()
            out_img = vedo.picture._get_img(tname)

        elif isinstance(tname, str):
            tu = vtk.vtkTexture()

            if "https://" in tname:
                try:
                    tname = vedo.file_io.download(tname, verbose=False)
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
                vedo.logger.error("in texture() supported files are only PNG, BMP or JPG")
                return self
            reader.SetFileName(fn)
            reader.Update()
            out_img = reader.GetOutput()

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

        if out_img:
            tu.SetInputData(out_img)
        tu.SetInterpolate(interpolate)
        tu.SetRepeat(repeat)
        tu.SetEdgeClamp(edge_clamp)

        self.property.SetColor(1, 1, 1)
        self.mapper.ScalarVisibilityOff()
        self.actor.SetTexture(tu)

        if seam_threshold is not None:
            tname = self.dataset.GetPointData().GetTCoords().GetName()
            grad = self.gradient(tname)
            ugrad, vgrad = np.split(grad, 2, axis=1)
            ugradm, vgradm = vedo.utils.mag2(ugrad), vedo.utils.mag2(vgrad)
            gradm = np.log(ugradm + vgradm)
            largegrad_ids = np.arange(len(grad))[gradm > seam_threshold * 4]
            uvmap = self.pointdata[tname]
            # collapse triangles that have large gradient
            new_points = self.vertices.copy()
            for f in self.cells:
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
            self.vertices = new_points

        self.dataset.Modified()
        self._texture = {
            "tname": tname,
            "tcoords": tcoords,
            "interpolate": interpolate,
            "repeat": repeat,
            "edge_clamp": edge_clamp,
            "scale": scale,
            "ushift": ushift,
            "vshift": vshift,
            "seam_threshold": seam_threshold
        }
        return self

    def compute_normals(self, points=True, cells=True, feature_angle=None, consistency=True):
        """
        Compute cell and vertex normals for the mesh.

        Arguments:
            points : (bool)
                do the computation for the vertices too
            cells : (bool)
                do the computation for the cells too
            feature_angle : (float)
                specify the angle that defines a sharp edge.
                If the difference in angle across neighboring polygons is greater than this value,
                the shared edge is considered "sharp" and it is split.
            consistency : (bool)
                turn on/off the enforcement of consistent polygon ordering.

        .. warning::
            If feature_angle is set to a float the Mesh can be modified, and it
            can have a different nr. of vertices from the original.
        """
        poly = self.dataset
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
        self.dataset.GetPointData().SetNormals(pdnorm.GetOutput().GetPointData().GetNormals())
        self.dataset.GetCellData().SetNormals(pdnorm.GetOutput().GetCellData().GetNormals())
        return self

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
        poly = self.dataset

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
        self._update(rev.GetOutput(), reset_locators=False)
        self.pipeline = OperationNode("reverse", parents=[self])
        return self

    def volume(self):
        """Get/set the volume occupied by mesh."""
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.dataset)
        mass.Update()
        return mass.GetVolume()

    def area(self):
        """
        Compute the surface area of mesh.
        The mesh must be triangular for this to work.
        See also `mesh.triangulate()`.
        """
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.dataset)
        mass.Update()
        return mass.GetSurfaceArea()

    def is_closed(self):
        """Return `True` if the mesh is watertight."""
        fe = vtk.vtkFeatureEdges()
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOn()
        fe.SetInputData(self.dataset)
        fe.Update()
        ne = fe.GetOutput().GetNumberOfCells()
        return not bool(ne)

    def is_manifold(self):
        """Return `True` if the mesh is manifold."""
        fe = vtk.vtkFeatureEdges()
        fe.BoundaryEdgesOff()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOn()
        fe.SetInputData(self.dataset)
        fe.Update()
        ne = fe.GetOutput().GetNumberOfCells()
        return not bool(ne)

    def non_manifold_faces(self, remove=True, tol="auto"):
        """
        Detect and (try to) remove non-manifold faces of a triangular mesh.

        Set `remove` to `False` to mark cells without removing them.
        Set `tol=0` for zero-tolerance, the result will be manifold but with holes.
        Set `tol>0` to cut off non-manifold faces, and try to recover the good ones.
        Set `tol="auto"` to make an automatic choice of the tolerance.
        """
        # mark original point and cell ids
        self.add_ids()
        toremove = self.boundaries(
            boundary_edges=False, non_manifold_edges=True, cell_edge=True, return_cell_ids=True
        )
        if len(toremove) == 0:
            return self

        points = self.vertices
        faces = self.cells
        centers = self.cell_centers

        copy = self.clone()
        copy.delete_cells(toremove).clean()
        copy.compute_normals(cells=False)
        normals = copy.normals()
        deltas, deltas_i = [], []

        for i in vedo.utils.progressbar(toremove, delay=3, title="recover faces"):
            pids = copy.closest_point(centers[i], n=3, return_point_id=True)
            norms = normals[pids]
            n = np.mean(norms, axis=0)
            dn = np.linalg.norm(n)
            if not dn:
                continue
            n = n / dn

            p0, p1, p2 = points[faces[i]][:3]
            v = np.cross(p1 - p0, p2 - p0)
            lv = np.linalg.norm(v)
            if not lv:
                continue
            v = v / lv

            cosa = 1 - np.dot(n, v)
            deltas.append(cosa)
            deltas_i.append(i)

        recover = []
        if len(deltas) > 0:
            mean_delta = np.mean(deltas)
            err_delta = np.std(deltas)
            txt = ""
            if tol == "auto":  # automatic choice
                tol = mean_delta / 5
                txt = f"\n Automatic tol. : {tol: .4f}"
            for i, cosa in zip(deltas_i, deltas):
                if cosa < tol:
                    recover.append(i)

            vedo.logger.info(
                f"\n --------- Non manifold faces ---------"
                f"\n Average tol.   : {mean_delta: .4f} +- {err_delta: .4f}{txt}"
                f"\n Removed faces  : {len(toremove)}"
                f"\n Recovered faces: {len(recover)}"
            )

        toremove = list(set(toremove) - set(recover))

        if not remove:
            mark = np.zeros(self.ncells, dtype=np.uint8)
            mark[recover] = 1
            mark[toremove] = 2
            self.celldata["NonManifoldCell"] = mark
        else:
            self.delete_cells(toremove)

        self.pipeline = OperationNode(
            "non_manifold_faces", parents=[self], comment=f"#cells {self.dataset.GetNumberOfCells()}"
        )
        return self

    def shrink(self, fraction=0.85):
        """Shrink the triangle polydata in the representation of the input mesh.

        Examples:
            - [shrink.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/shrink.py)

            ![](https://vedo.embl.es/images/basic/shrink.png)
        """
        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputData(self.dataset)
        shrink.SetShrinkFactor(fraction)
        shrink.Update()
        self.point_locator = None
        self.cell_locator = None
        self._update(shrink.GetOutput())
        self.pipeline = OperationNode("shrink", parents=[self])
        return self

    def stretch(self, q1, q2):
        """
        Stretch mesh between points `q1` and `q2`.

        Examples:
            - [aspring1.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/aspring1.py)

            ![](https://vedo.embl.es/images/simulations/50738955-7e891800-11d9-11e9-85cd-02bd4f3f13ea.gif)

        .. note::
            for `Mesh` objects, two vectors `mesh.base`, and `mesh.top` must be defined.
        """
        if self.base is None:
            vedo.logger.error("in stretch() must define vectors mesh.base and mesh.top at creation")
            raise RuntimeError()

        p1, p2 = self.base, self.top
        q1, q2, z = np.asarray(q1), np.asarray(q2), np.array([0, 0, 1])
        a = p2 - p1
        b = q2 - q1
        plength = np.linalg.norm(a)
        qlength = np.linalg.norm(b)
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.Translate(-p1)
        cosa = np.dot(a, z) / plength
        n = np.cross(a, z)
        if np.linalg.norm(n):
            T.RotateWXYZ(np.rad2deg(np.arccos(cosa)), n)
        T.Scale(1, 1, qlength / plength)

        cosa = np.dot(b, z) / qlength
        n = np.cross(b, z)
        if np.linalg.norm(n):
            T.RotateWXYZ(-np.rad2deg(np.arccos(cosa)), n)
        else:
            if np.dot(b, z) < 0:
                T.RotateWXYZ(180, [1, 0, 0])

        T.Translate(q1)

        self.apply_transform(T)
        return self


    def cap(self, return_cap=False):
        """
        Generate a "cap" on a clipped mesh, or caps sharp edges.

        Examples:
            - [cut_and_cap.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/cut_and_cap.py)

            ![](https://vedo.embl.es/images/advanced/cutAndCap.png)

        See also: `join()`, `join_segments()`, `slice()`.
        """
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(self.dataset)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ManifoldEdgesOff()
        fe.Update()

        stripper = vtk.vtkStripper()
        stripper.SetInputData(fe.GetOutput())
        stripper.JoinContiguousSegmentsOn()
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
            m.pipeline = OperationNode(
                "cap", parents=[self],
                comment=f"#pts {m.dataset.GetNumberOfPoints()}"
            )
            return m

        polyapp = vtk.vtkAppendPolyData()
        polyapp.AddInputData(self.dataset)
        polyapp.AddInputData(tf.GetOutput())
        polyapp.Update()

        self._update(polyapp.GetOutput())
        self.clean()

        self.pipeline = OperationNode(
            "capped", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

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

        Arguments:
            polys : (bool)
                polygonal segments will be joined if they are contiguous
            reset : (bool)
                reset points ordering

        Warning:
            If triangle strips or polylines exist in the input data
            they will be passed through to the output data.
            This filter will only construct triangle strips if triangle polygons
            are available; and will only construct polylines if lines are available.

        Example:
            ```python
            from vedo import *
            c1 = Cylinder(pos=(0,0,0), r=2, height=3, axis=(1,.0,0), alpha=.1).triangulate()
            c2 = Cylinder(pos=(0,0,2), r=1, height=2, axis=(0,.3,1), alpha=.1).triangulate()
            intersect = c1.intersect_with(c2).join(reset=True)
            spline = Spline(intersect).c('blue').lw(5)
            show(c1, c2, spline, intersect.labels('id'), axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/line_join.png)
        """
        sf = vtk.vtkStripper()
        sf.SetPassThroughCellIds(True)
        sf.SetPassThroughPointIds(True)
        sf.SetJoinContiguousSegments(polys)
        sf.SetInputData(self.dataset)
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
        else:
            poly = sf.GetOutput()

        self._update(poly)

        self.pipeline = OperationNode(
            "join", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def join_segments(self, closed=True, tol=1e-03):
        """
        Join line segments into contiguous lines.
        Useful to call with `triangulate()` method.

        Returns:
            list of `shapes.Lines`

        Example:
            ```python
            from vedo import *
            msh = Torus().alpha(0.1).wireframe()
            intersection = msh.intersect_with_plane(normal=[1,1,1]).c('purple5')
            slices = [s.triangulate() for s in intersection.join_segments()]
            show(msh, intersection, merge(slices), axes=1, viewup='z')
            ```
            ![](https://vedo.embl.es/images/feats/join_segments.jpg)
        """
        vlines = []
        for ipiece, outline in enumerate(self.split(must_share_edge=False)):

            outline.clean()
            pts = outline.vertices
            if len(pts) < 3:
                continue
            avesize = outline.average_size()
            lines = outline.lines()
            # print("---lines", lines, "in piece", ipiece)
            tol = avesize / pts.shape[0] * tol

            k = 0
            joinedpts = [pts[k]]
            for _ in range(len(pts)):
                pk = pts[k]
                for j, line in enumerate(lines):

                    id0, id1 = line[0], line[-1]
                    p0, p1 = pts[id0], pts[id1]

                    if np.linalg.norm(p0 - pk) < tol:
                        n = len(line)
                        for m in range(1, n):
                            joinedpts.append(pts[line[m]])
                        # joinedpts.append(p1)
                        k = id1
                        lines.pop(j)
                        break

                    elif np.linalg.norm(p1 - pk) < tol:
                        n = len(line)
                        for m in reversed(range(0, n - 1)):
                            joinedpts.append(pts[line[m]])
                        # joinedpts.append(p0)
                        k = id0
                        lines.pop(j)
                        break

            if len(joinedpts) > 1:
                newline = vedo.shapes.Line(joinedpts, closed=closed)
                newline.clean()
                newline.actor.SetProperty(self.property)
                newline.property = self.property
                newline.pipeline = OperationNode(
                    "join_segments",
                    parents=[self],
                    comment=f"#pts {newline.dataset.GetNumberOfPoints()}",
                )
                vlines.append(newline)

        return vlines

    def slice(self, origin=(0, 0, 0), normal=(1, 0, 0)):
        """
        Slice a mesh with a plane and fill the contour.

        Example:
            ```python
            from vedo import *
            msh = Mesh(dataurl+"bunny.obj").alpha(0.1).wireframe()
            mslice = msh.slice(normal=[0,1,0.3], origin=[0,0.16,0])
            mslice.c('purple5')
            show(msh, mslice, axes=1)
            ```
            ![](https://vedo.embl.es/images/feats/mesh_slice.jpg)

        See also: `join()`, `join_segments()`, `cap()`, `cut_with_plane()`.
        """
        intersection = self.intersect_with_plane(origin=origin, normal=normal)
        slices = [s.triangulate() for s in intersection.join_segments()]
        mslices = vedo.pointcloud.merge(slices)
        if mslices:
            mslices.name = "MeshSlice"
            mslices.pipeline = OperationNode("slice", parents=[self], comment=f"normal = {normal}")
        return mslices

    def triangulate(self, verts=True, lines=True):
        """
        Converts mesh polygons into triangles.

        If the input mesh is only made of 2D lines (no faces) the output will be a triangulation
        that fills the internal area. The contours may be concave, and may even contain holes,
        i.e. a contour may contain an internal contour winding in the opposite
        direction to indicate that it is a hole.

        Arguments:
            verts : (bool)
                if True, break input vertex cells into individual vertex cells
                (one point per cell). If False, the input vertex cells will be ignored.
            lines : (bool)
                if True, break input polylines into line segments.
                If False, input lines will be ignored and the output will have no lines.
        """
        if self.dataset.GetNumberOfPolys() or self.dataset.GetNumberOfStrips():
            # print("vtkTriangleFilter")
            tf = vtk.vtkTriangleFilter()
            tf.SetPassLines(lines)
            tf.SetPassVerts(verts)

        elif self.dataset.GetNumberOfLines():
            # print("vtkContourTriangulator")
            tf = vtk.vtkContourTriangulator()
            tf.TriangulationErrorDisplayOn()

        else:
            vedo.logger.debug("input in triangulate() seems to be void! Skip.")
            return self

        tf.SetInputData(self.dataset)
        tf.Update()
        self._update(tf.GetOutput(), reset_locators=False)
        self.lw(0).lighting("default").pickable()

        self.pipeline = OperationNode(
            "triangulate", parents=[self], comment=f"#cells {self.dataset.GetNumberOfCells()}"
        )
        return self

    def compute_cell_area(self, name="Area"):
        """Add to this mesh a cell data array containing the areas of the polygonal faces"""
        csf = vtk.vtkCellSizeFilter()
        csf.SetInputData(self.dataset)
        csf.SetComputeArea(True)
        csf.SetComputeVolume(False)
        csf.SetComputeLength(False)
        csf.SetComputeVertexCount(False)
        csf.SetAreaArrayName(name)
        csf.Update()
        self.dataset.GetCellData().AddArray(csf.GetOutput().GetCellData().GetArray(name))
        return self

    def compute_cell_vertex_count(self, name="VertexCount"):
        """Add to this mesh a cell data array containing the nr of vertices
        that a polygonal face has."""
        csf = vtk.vtkCellSizeFilter()
        csf.SetInputData(self.dataset)
        csf.SetComputeArea(False)
        csf.SetComputeVolume(False)
        csf.SetComputeLength(False)
        csf.SetComputeVertexCount(True)
        csf.SetVertexCountArrayName(name)
        csf.Update()
        self.dataset.GetCellData().AddArray(csf.GetOutput().GetCellData().GetArray(name))
        return self

    def compute_quality(self, metric=6):
        """
        Calculate metrics of quality for the elements of a triangular mesh.
        This method adds to the mesh a cell array named "Quality".
        See class [vtkMeshQuality](https://vtk.org/doc/nightly/html/classvtkMeshQuality.html)
        for explanation.

        Arguments:
            metric : (int)
                type of available estimators are:
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

        Examples:
            - [meshquality.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/meshquality.py)

            ![](https://vedo.embl.es/images/advanced/meshquality.png)
        """
        qf = vtk.vtkMeshQuality()
        qf.SetInputData(self.dataset)
        qf.SetTriangleQualityMeasure(metric)
        qf.SaveCellQualityOn()
        qf.Update()
        self._update(qf.GetOutput(), reset_locators=False)
        self.pipeline = OperationNode("compute_quality", parents=[self])
        return self

    def check_validity(self, tol=0):
        """
        Return an array of possible problematic faces following this convention:
        - Valid               =  0
        - WrongNumberOfPoints =  1
        - IntersectingEdges   =  2
        - IntersectingFaces   =  4
        - NoncontiguousEdges  =  8
        - Nonconvex           = 10
        - OrientedIncorrectly = 20

        Arguments:
            tol : (float)
                value is used as an epsilon for floating point
                equality checks throughout the cell checking process.
        """
        vald = vtk.vtkCellValidator()
        if tol:
            vald.SetTolerance(tol)
        vald.SetInputData(self.dataset)
        vald.Update()
        varr = vald.GetOutput().GetCellData().GetArray("ValidityState")
        return vtk2numpy(varr)

    def compute_curvature(self, method=0):
        """
        Add scalars to `Mesh` that contains the curvature calculated in three different ways.

        Variable `method` can be:
        - 0 = gaussian
        - 1 = mean curvature
        - 2 = max curvature
        - 3 = min curvature

        Example:
            ```python
            from vedo import Torus
            Torus().compute_curvature().add_scalarbar().show(axes=1).close()
            ```
            ![](https://user-images.githubusercontent.com/32848391/51934810-c2e88c00-2404-11e9-8e7e-ca0b7984bbb7.png)
        """
        curve = vtk.vtkCurvatures()
        curve.SetInputData(self.dataset)
        curve.SetCurvatureType(method)
        curve.Update()
        self._update(curve.GetOutput(), reset_locators=False)
        self.mapper.ScalarVisibilityOn()
        return self

    def compute_elevation(self, low=(0, 0, 0), high=(0, 0, 1), vrange=(0, 1)):
        """
        Add to `Mesh` a scalar array that contains distance along a specified direction.

        Arguments:
            low : (list)
                one end of the line (small scalar values)
            high : (list)
                other end of the line (large scalar values)
            vrange : (list)
                set the range of the scalar

        Example:
            ```python
            from vedo import Sphere
            s = Sphere().compute_elevation(low=(0,0,0), high=(1,1,1))
            s.add_scalarbar().show(axes=1).close()
            ```
            ![](https://user-images.githubusercontent.com/32848391/68478872-3986a580-0231-11ea-8245-b68a683aa295.png)
        """
        ef = vtk.vtkElevationFilter()
        ef.SetInputData(self.dataset)
        ef.SetLowPoint(low)
        ef.SetHighPoint(high)
        ef.SetScalarRange(vrange)
        ef.Update()
        self._update(ef.GetOutput(), reset_locators=False)
        self.mapper.ScalarVisibilityOn()
        return self

    def subdivide(self, n=1, method=0, mel=None):
        """
        Increase the number of vertices of a surface mesh.

        Arguments:
            n : (int)
                number of subdivisions.
            method : (int)
                Loop(0), Linear(1), Adaptive(2), Butterfly(3), Centroid(4)
            mel : (float)
                Maximum Edge Length (applicable to Adaptive method only).
        """
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputData(self.dataset)
        triangles.Update()
        originalMesh = triangles.GetOutput()
        if method == 0:
            sdf = vtk.vtkLoopSubdivisionFilter()
        elif method == 1:
            sdf = vtk.vtkLinearSubdivisionFilter()
        elif method == 2:
            sdf = vtk.vtkAdaptiveSubdivisionFilter()
            if mel is None:
                mel = self.diagonal_size() / np.sqrt(self.dataset.GetNumberOfPoints()) / n
            sdf.SetMaximumEdgeLength(mel)
        elif method == 3:
            sdf = vtk.vtkButterflySubdivisionFilter()
        elif method == 4:
            sdf = vtk.vtkDensifyPolyData()
        else:
            vedo.logger.error(f"in subdivide() unknown method {method}")
            raise RuntimeError()

        if method != 2:
            sdf.SetNumberOfSubdivisions(n)

        sdf.SetInputData(originalMesh)
        sdf.Update()

        self._update(sdf.GetOutput())

        self.pipeline = OperationNode(
            "subdivide", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def decimate(self, fraction=0.5, n=None, method="quadric", boundaries=False):
        """
        Downsample the number of vertices in a mesh to `fraction`.

        Arguments:
            fraction : (float)
                the desired target of reduction.
            n : (int)
                the desired number of final points (`fraction` is recalculated based on it).
            method : (str)
                can be either 'quadric' or 'pro'. In the first case triagulation
                will look like more regular, irrespective of the mesh original curvature.
                In the second case triangles are more irregular but mesh is more precise on more
                curved regions.
            boundaries : (bool)
                in "pro" mode decide whether to leave boundaries untouched or not

        .. note:: Setting `fraction=0.1` leaves 10% of the original number of vertices
        """
        poly = self.dataset
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

        self._update(decimate.GetOutput())

        self.pipeline = OperationNode(
            "decimate", parents=[self],
            comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def collapse_edges(self, distance, iterations=1):
        """Collapse mesh edges so that are all above distance."""
        self.clean()
        x0, x1, y0, y1, z0, z1 = self.bounds()
        fs = min(x1 - x0, y1 - y0, z1 - z0) / 10
        d2 = distance * distance
        if distance > fs:
            vedo.logger.error(f"distance parameter is too large, should be < {fs}, skip!")
            return self
        for _ in range(iterations):
            medges = self.edges()
            pts = self.vertices
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

            self.vertices = newpts
            self.clean()
        self.compute_normals()

        self.pipeline = OperationNode(
            "collapse_edges", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def smooth(self, niter=15, pass_band=0.1, edge_angle=15, feature_angle=60, boundary=False):
        """
        Adjust mesh point positions using the `Windowed Sinc` function interpolation kernel.

        Arguments:
            niter : (int)
                number of iterations.
            pass_band : (float)
                set the pass_band value for the windowed sinc filter.
            edge_angle : (float)
                edge angle to control smoothing along edges (either interior or boundary).
            feature_angle : (float)
                specifies the feature angle for sharp edge identification.
            boundary : (bool)
                specify if boundary should also be smoothed or kept unmodified

        Examples:
            - [mesh_smoother1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/mesh_smoother1.py)

            ![](https://vedo.embl.es/images/advanced/mesh_smoother2.png)
        """
        poly = self.dataset
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        smf = vtk.vtkWindowedSincPolyDataFilter()
        smf.SetInputData(cl.GetOutput())
        smf.SetNumberOfIterations(niter)
        smf.SetEdgeAngle(edge_angle)
        smf.SetFeatureAngle(feature_angle)
        smf.SetPassBand(pass_band)
        smf.NormalizeCoordinatesOn()
        smf.NonManifoldSmoothingOn()
        smf.FeatureEdgeSmoothingOn()
        smf.SetBoundarySmoothing(boundary)
        smf.Update()

        self._update(smf.GetOutput())

        self.pipeline = OperationNode(
            "smooth", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self


    def fill_holes(self, size=None):
        """
        Identifies and fills holes in input mesh.
        Holes are identified by locating boundary edges, linking them together into loops,
        and then triangulating the resulting loops.

        Arguments:
            size : (float)
                Approximate limit to the size of the hole that can be filled. The default is None.

        Examples:
            - [fillholes.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/fillholes.py)
        """
        fh = vtk.vtkFillHolesFilter()
        if not size:
            mb = self.diagonal_size()
            size = mb / 10
        fh.SetHoleSize(size)
        fh.SetInputData(self.dataset)
        fh.Update()

        self._update(fh.GetOutput())

        self.pipeline = OperationNode(
            "fill_holes", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def is_inside(self, point, tol=1e-05):
        """Return True if point is inside a polydata closed surface."""
        poly = self
        points = vtk.vtkPoints()
        points.InsertNextPoint(point)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        sep = vtk.vtkSelectEnclosedPoints()
        sep.SetTolerance(tol)
        sep.CheckSurfaceOff()
        sep.SetInputData(poly)
        sep.SetSurfaceData(poly)
        sep.Update()
        return sep.IsInside(0)

    def inside_points(self, pts, invert=False, tol=1e-05, return_ids=False):
        """
        Return the point cloud that is inside mesh surface as a new Points object.

        If return_ids is True a list of IDs is returned and in addition input points
        are marked by a pointdata array named "IsInside".

        Example:
            `print(pts.pointdata["IsInside"])`

        Examples:
            - [pca_ellipsoid.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/pca_ellipsoid.py)

            ![](https://vedo.embl.es/images/basic/pca.png)
        """
        if isinstance(pts, Points):
            poly = pts.dataset
            ptsa = pts.vertices
        else:
            ptsa = np.asarray(pts)
            vpoints = vtk.vtkPoints()
            vpoints.SetData(numpy2vtk(ptsa, dtype=np.float32))
            poly = vtk.vtkPolyData()
            poly.SetPoints(vpoints)

        sep = vtk.vtkSelectEnclosedPoints()
        # sep = vtk.vtkExtractEnclosedPoints()
        sep.SetTolerance(tol)
        sep.SetInputData(poly)
        sep.SetSurfaceData(self.dataset)
        sep.SetInsideOut(invert)
        sep.Update()

        varr = sep.GetOutput().GetPointData().GetArray("SelectedPoints")
        mask = vtk2numpy(varr).astype(bool)
        ids = np.array(range(len(ptsa)), dtype=int)[mask]

        if isinstance(pts, Points):
            varr.SetName("IsInside")
            pts.dataset.GetPointData().AddArray(varr)

        if return_ids:
            return ids

        pcl = Points(ptsa[ids])
        pcl.name = "InsidePoints"

        pcl.pipeline = OperationNode(
            "inside_points", parents=[self, ptsa],
            comment=f"#pts {pcl.dataset.GetNumberOfPoints()}"
        )
        return pcl

    def boundaries(
        self,
        boundary_edges=True,
        manifold_edges=False,
        non_manifold_edges=False,
        feature_angle=None,
        return_point_ids=False,
        return_cell_ids=False,
        cell_edge=False,
    ):
        """
        Return the boundary lines of an input mesh.
        Check also `vedo.base.BaseActor.mark_boundaries()` method.

        Arguments:
            boundary_edges : (bool)
                Turn on/off the extraction of boundary edges.
            manifold_edges : (bool)
                Turn on/off the extraction of manifold edges.
            non_manifold_edges : (bool)
                Turn on/off the extraction of non-manifold edges.
            feature_angle : (bool)
                Specify the min angle btw 2 faces for extracting edges.
            return_point_ids : (bool)
                return a numpy array of point indices
            return_cell_ids : (bool)
                return a numpy array of cell indices
            cell_edge : (bool)
                set to `True` if a cell need to share an edge with
                the boundary line, or `False` if a single vertex is enough

        Examples:
            - [boundaries.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/boundaries.py)

            ![](https://vedo.embl.es/images/basic/boundaries.png)
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
            idf.SetInputData(self.dataset)
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
                n = 1 if cell_edge else 0
                inface = []
                for i, face in enumerate(self.cells):
                    # isin = np.any([vtx in npid for vtx in face])
                    isin = 0
                    for vtx in face:
                        isin += int(vtx in npid)
                        if isin > n:
                            break
                    if isin > n:
                        inface.append(i)
                return np.array(inface).astype(int)

            return self

        else:

            fe.SetInputData(self.dataset)
            fe.Update()
            msh = Mesh(fe.GetOutput(), c="p").lw(5).lighting("off")

            msh.pipeline = OperationNode(
                "boundaries",
                parents=[self],
                shape="octagon",
                comment=f"#pts {msh.dataset.GetNumberOfPoints()}",
            )
            return msh

    def imprint(self, loopline, tol=0.01):
        """
        Imprint the contact surface of one object onto another surface.

        Arguments:
            loopline : vedo.shapes.Line
                a Line object to be imprinted onto the mesh.
            tol : (float), optional
                projection tolerance which controls how close the imprint
                surface must be to the target.

        Example:
            ```python
            from vedo import *
            grid = Grid()#.triangulate()
            circle = Circle(r=0.3, res=24).pos(0.11,0.12)
            line = Line(circle, closed=True, lw=4, c='r4')
            grid.imprint(line)
            show(grid, line, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/imprint.png)
        """
        loop = vtk.vtkContourLoopExtraction()
        loop.SetInputData(loopline)
        loop.Update()

        clean_loop = vtk.vtkCleanPolyData()
        clean_loop.SetInputData(loop.GetOutput())
        clean_loop.Update()

        imp = vtk.vtkImprintFilter()
        imp.SetTargetData(self.dataset)
        imp.SetImprintData(clean_loop.GetOutput())
        imp.SetTolerance(tol)
        imp.BoundaryEdgeInsertionOn()
        imp.TriangulateOutputOn()
        imp.Update()

        self._update(imp.GetOutput())

        self.pipeline = OperationNode(
            "imprint", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def connected_vertices(self, index):
        """Find all vertices connected to an input vertex specified by its index.

        Examples:
            - [connected_vtx.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/connected_vtx.py)

            ![](https://vedo.embl.es/images/basic/connVtx.png)
        """
        poly = self.dataset

        cell_idlist = vtk.vtkIdList()
        poly.GetPointCells(index, cell_idlist)

        idxs = []
        for i in range(cell_idlist.GetNumberOfIds()):
            point_idlist = vtk.vtkIdList()
            poly.GetCellPoints(cell_idlist.GetId(i), point_idlist)
            for j in range(point_idlist.GetNumberOfIds()):
                idj = point_idlist.GetId(j)
                if idj == index:
                    continue
                if idj in idxs:
                    continue
                idxs.append(idj)

        return idxs

    def connected_cells(self, index, return_ids=False):
        """Find all cellls connected to an input vertex specified by its index."""

        # Find all cells connected to point index
        dpoly = self.dataset
        idlist = vtk.vtkIdList()
        dpoly.GetPointCells(index, idlist)

        ids = vtk.vtkIdTypeArray()
        ids.SetNumberOfComponents(1)
        rids = []
        for k in range(idlist.GetNumberOfIds()):
            cid = idlist.GetId(k)
            ids.InsertNextValue(cid)
            rids.append(int(cid))
        if return_ids:
            return rids

        selection_node = vtk.vtkSelectionNode()
        selection_node.SetFieldType(vtk.vtkSelectionNode.CELL)
        selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
        selection_node.SetSelectionList(ids)
        selection = vtk.vtkSelection()
        selection.AddNode(selection_node)
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, dpoly)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(extractSelection.GetOutput())
        gf.Update()
        return Mesh(gf.GetOutput()).lw(1)

    def silhouette(self, direction=None, border_edges=True, feature_angle=False):
        """
        Return a new line `Mesh` which corresponds to the outer `silhouette`
        of the input as seen along a specified `direction`, this can also be
        a `vtkCamera` object.

        Arguments:
            direction : (list)
                viewpoint direction vector.
                If *None* this is guessed by looking at the minimum
                of the sides of the bounding box.
            border_edges : (bool)
                enable or disable generation of border edges
            feature_angle : (float)
                minimal angle for sharp edges detection.
                If set to `False` the functionality is disabled.

        Examples:
            - [silhouette1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/silhouette1.py)

            ![](https://vedo.embl.es/images/basic/silhouette1.png)
        """
        sil = vtk.vtkPolyDataSilhouette()
        sil.SetInputData(self.dataset)
        sil.SetBorderEdges(border_edges)
        if feature_angle is False:
            sil.SetEnableFeatureAngle(0)
        else:
            sil.SetEnableFeatureAngle(1)
            sil.SetFeatureAngle(feature_angle)

        if direction is None and vedo.plotter_instance and vedo.plotter_instance.camera:
            sil.SetCamera(vedo.plotter_instance.camera)
            m = Mesh()
            m.mapper.SetInputConnection(sil.GetOutputPort())

        elif isinstance(direction, vtk.vtkCamera):
            sil.SetCamera(direction)
            m = Mesh()
            m.mapper.SetInputConnection(sil.GetOutputPort())

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
        m.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        m.pipeline = OperationNode("silhouette", parents=[self])
        return m

    def isobands(self, n=10, vmin=None, vmax=None):
        """
        Return a new `Mesh` representing the isobands of the active scalars.
        This is a new mesh where the scalar is now associated to cell faces and
        used to colorize the mesh.

        Arguments:
            n : (int)
                number of isobands in the range
            vmin : (float)
                minimum of the range
            vmax : (float)
                maximum of the range

        Examples:
            - [isolines.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/isolines.py)
        """
        r0, r1 = self.dataset.GetScalarRange()
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
        lut = self.mapper.GetLookupTable()
        labels = []
        for b in bands:
            labels.append("{:4.2f}".format(b[1]))
        values = vtk.vtkVariantArray()
        for la in labels:
            values.InsertNextValue(vtk.vtkVariant(la))
        for i in range(values.GetNumberOfTuples()):
            lut.SetAnnotation(i, values.GetValue(i).ToString())

        bcf = vtk.vtkBandedPolyDataContourFilter()
        bcf.SetInputData(self.dataset)
        # Use either the minimum or maximum value for each band.
        for i, band in enumerate(bands):
            bcf.SetValue(i, band[2])
        # We will use an indexed lookup table.
        bcf.SetScalarModeToIndex()
        bcf.GenerateContourEdgesOff()
        bcf.Update()
        bcf.GetOutput().GetCellData().GetScalars().SetName("IsoBands")
        m1 = Mesh(bcf.GetOutput()).compute_normals(cells=True)
        m1.mapper.SetLookupTable(lut)

        m1.pipeline = OperationNode("isobands", parents=[self])
        return m1

    def isolines(self, n=10, vmin=None, vmax=None):
        """
        Return a new `Mesh` representing the isolines of the active scalars.

        Arguments:
            n : (int)
                number of isolines in the range
            vmin : (float)
                minimum of the range
            vmax : (float)
                maximum of the range

        Examples:
            - [isolines.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/isolines.py)

            ![](https://vedo.embl.es/images/pyplot/isolines.png)
        """
        bcf = vtk.vtkContourFilter()
        bcf.SetInputData(self.dataset)
        r0, r1 = self.dataset.GetScalarRange()
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
        msh.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        msh.pipeline = OperationNode("isolines", parents=[self])
        return msh

    def extrude(self, zshift=1, rotation=0, dr=0, cap=True, res=1):
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

        Examples:
            - [extrude.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/extrude.py)

            ![](https://vedo.embl.es/images/basic/extrude.png)
        """
        if is_sequence(zshift):
            # ms = [] # todo
            # poly0 = self.clone()
            # for i in range(len(zshift)-1):
            #     rf = vtk.vtkRotationalExtrusionFilter()
            #     rf.SetInputData(poly0)
            #     rf.SetResolution(res)
            #     rf.SetCapping(0)
            #     rf.SetAngle(rotation)
            #     rf.SetTranslation(zshift)
            #     rf.SetDeltaRadius(dR)
            #     rf.Update()
            #     poly1 = rf.GetOutput()
            return self

        rf = vtk.vtkRotationalExtrusionFilter()
        # rf = vtk.vtkLinearExtrusionFilter()
        rf.SetInputData(self.dataset)  # must not be transformed
        rf.SetResolution(res)
        rf.SetCapping(cap)
        rf.SetAngle(rotation)
        rf.SetTranslation(zshift)
        rf.SetDeltaRadius(dr)
        rf.Update()

        m = Mesh(rf.GetOutput(), c=self.c(), alpha=self.alpha())
        prop = vtk.vtkProperty()
        prop.DeepCopy(self.property)
        m.actor.SetProperty(prop)
        m.property = prop

        m.compute_normals(cells=False).flat().lighting("default")

        m.pipeline = OperationNode(
            "extrude", parents=[self],
            comment=f"#pts {m.dataset.GetNumberOfPoints()}"
        )
        return m

    def split(self, maxdepth=1000, flag=False, must_share_edge=False, sort_by_area=True):
        """
        Split a mesh by connectivity and order the pieces by increasing area.

        Arguments:
            maxdepth : (int)
                only consider this maximum number of mesh parts.
            flag : (bool)
                if set to True return the same single object,
                but add a "RegionId" array to flag the mesh subparts
            must_share_edge : (bool)
                if True, mesh regions that only share single points will be split.
            sort_by_area : (bool)
                if True, sort the mesh parts by decreasing area.

        Examples:
            - [splitmesh.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/splitmesh.py)

            ![](https://vedo.embl.es/images/advanced/splitmesh.png)
        """
        pd = self.dataset
        if must_share_edge:
            if pd.GetNumberOfPolys() == 0:
                vedo.logger.warning("in split(): no polygons found. Skip.")
                return [self]
            cf = vtk.vtkPolyDataEdgeConnectivityFilter()
            cf.BarrierEdgesOff()
        else:
            cf = vtk.vtkPolyDataConnectivityFilter()

        cf.SetInputData(pd)
        cf.SetExtractionModeToAllRegions()
        cf.SetColorRegions(True)
        cf.Update()
        out = cf.GetOutput()

        if not out.GetNumberOfPoints():
            return [self]

        if flag:
            self.pipeline = OperationNode("split mesh", parents=[self])
            self._update(out)
            return self

        a = Mesh(out)
        if must_share_edge:
            arr = a.celldata["RegionId"]
            on = "cells"
        else:
            arr = a.pointdata["RegionId"]
            on = "points"

        alist = []
        for t in range(max(arr) + 1):
            if t == maxdepth:
                break
            suba = a.clone().threshold("RegionId", t, t, on=on)
            if sort_by_area:
                area = suba.area()
            else:
                area = 0  # dummy
            alist.append([suba, area])

        if sort_by_area:
            alist.sort(key=lambda x: x[1])
            alist.reverse()

        blist = []
        for i, l in enumerate(alist):
            l[0].color(i + 1).phong()
            l[0].mapper.ScalarVisibilityOff()
            blist.append(l[0])
            if i < 10:
                l[0].pipeline = OperationNode(
                    f"split mesh {i}",
                    parents=[self],
                    comment=f"#pts {l[0].dataset.GetNumberOfPoints()}",
                )
        return blist


    def extract_largest_region(self):
        """
        Extract the largest connected part of a mesh and discard all the smaller pieces.

        Examples:
            - [largestregion.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/largestregion.py)
        """
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetExtractionModeToLargestRegion()
        conn.ScalarConnectivityOff()
        conn.SetInputData(self.dataset)
        conn.Update()
        m = Mesh(conn.GetOutput())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.property)
        m.SetProperty(pr)
        m.property = pr
        vis = self.mapper.GetScalarVisibility()
        m.mapper.SetScalarVisibility(vis)

        m.pipeline = OperationNode(
            "extract_largest_region", parents=[self],
            comment=f"#pts {m.dataset.GetNumberOfPoints()}"
        )
        return m

    def boolean(self, operation, mesh2, method=0, tol=None):
        """Volumetric union, intersection and subtraction of surfaces.

        Use `operation` for the allowed operations `['plus', 'intersect', 'minus']`.

        Two possible algorithms are available by changing `method`.

        Example:
            - [boolean.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/boolean.py)

            ![](https://vedo.embl.es/images/basic/boolean.png)
        """
        if method == 0:
            bf = vtk.vtkBooleanOperationPolyDataFilter()
        elif method == 1:
            bf = vtk.vtkLoopBooleanPolyDataFilter()
        else:
            raise ValueError(f"Unknown method={method}")

        poly1 = self.compute_normals().dataset
        poly2 = mesh2.compute_normals().dataset

        if operation.lower() in ("plus", "+"):
            bf.SetOperationToUnion()
        elif operation.lower() == "intersect":
            bf.SetOperationToIntersection()
        elif operation.lower() in ("minus", "-"):
            bf.SetOperationToDifference()

        if tol:
            bf.SetTolerance(tol)

        bf.SetInputData(0, poly1)
        bf.SetInputData(1, poly2)
        bf.Update()

        msh = Mesh(bf.GetOutput(), c=None)
        msh.flat()
        msh.name = self.name + operation + mesh2.name

        msh.pipeline = OperationNode(
            "boolean " + operation,
            parents=[self, mesh2],
            shape="cylinder",
            comment=f"#pts {msh.dataset.GetNumberOfPoints()}",
        )
        return msh

    def intersect_with(self, mesh2, tol=1e-06):
        """
        Intersect this Mesh with the input surface to return a set of lines.

        Examples:
            - [surf_intersect.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/surf_intersect.py)

                ![](https://vedo.embl.es/images/basic/surfIntersect.png)
        """
        bf = vtk.vtkIntersectionPolyDataFilter()
        bf.SetGlobalWarningDisplay(0)
        poly1 = self.dataset
        poly2 = mesh2.dataset
        bf.SetTolerance(tol)
        bf.SetInputData(0, poly1)
        bf.SetInputData(1, poly2)
        bf.Update()
        msh = Mesh(bf.GetOutput(), c="k", alpha=1).lighting("off")
        msh.property.SetLineWidth(3)
        msh.name = "SurfaceIntersection"
        msh.pipeline = OperationNode(
            "intersect_with", parents=[self, mesh2], comment=f"#pts {msh.npoints}"
        )
        return msh

    def intersect_with_line(self, p0, p1=None, return_ids=False, tol=0):
        """
        Return the list of points intersecting the mesh
        along the segment defined by two points `p0` and `p1`.

        Use `return_ids` to return the cell ids along with point coords

        Example:
            ```python
            from vedo import *
            s = Spring()
            pts = s.intersect_with_line([0,0,0], [1,0.1,0])
            ln = Line([0,0,0], [1,0.1,0], c='blue')
            ps = Points(pts, r=10, c='r')
            show(s, ln, ps, bg='white').close()
            ```
            ![](https://user-images.githubusercontent.com/32848391/55967065-eee08300-5c79-11e9-8933-265e1bab9f7e.png)
        """
        if isinstance(p0, Points):
            p0, p1 = p0.vertices

        if not self.line_locator:
            self.line_locator = vtk.vtkOBBTree()
            self.line_locator.SetDataSet(self.dataset)
            if not tol:
                tol = mag(np.asarray(p1) - np.asarray(p0)) / 10000
            self.line_locator.SetTolerance(tol)
            self.line_locator.BuildLocator()

        vpts = vtk.vtkPoints()
        idlist = vtk.vtkIdList()
        self.line_locator.IntersectWithLine(p0, p1, vpts, idlist)
        pts = []
        for i in range(vpts.GetNumberOfPoints()):
            intersection = [0, 0, 0]
            vpts.GetPoint(i, intersection)
            pts.append(intersection)
        pts = np.array(pts)

        if return_ids:
            pts_ids = []
            for i in range(idlist.GetNumberOfIds()):
                cid = idlist.GetId(i)
                pts_ids.append(cid)
            return (pts, np.array(pts_ids).astype(np.uint32))

        return pts

    def intersect_with_plane(self, origin=(0, 0, 0), normal=(1, 0, 0)):
        """
        Intersect this Mesh with a plane to return a set of lines.

        Example:
            ```python
            from vedo import *
            sph = Sphere()
            mi = sph.clone().intersect_with_plane().join()
            print(mi.lines())
            show(sph, mi, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/intersect_plane.png)
        """
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        cutter = vtk.vtkPolyDataPlaneCutter()
        cutter.SetInputData(self.dataset)
        cutter.SetPlane(plane)
        cutter.InterpolateAttributesOn()
        cutter.ComputeNormalsOff()
        cutter.Update()

        msh = Mesh(cutter.GetOutput(), "k", 1).lighting("off")
        msh.GetProperty().SetLineWidth(3)
        msh.name = "PlaneIntersection"

        msh.pipeline = OperationNode(
            "intersect_with_plan", parents=[self],
            comment=f"#pts {msh.dataset.GetNumberOfPoints()}"
        )
        return msh

    # def intersect_with_multiplanes(self, origins, normals): ## WRONG
    #     """
    #     Generate a set of lines from cutting a mesh in n intervals
    #     between a minimum and maximum distance from a plane of given origin and normal.

    #     Arguments:
    #         origin : (list)
    #             the point of the cutting plane
    #         normal : (list)
    #             normal vector to the cutting plane
    #         n : (int)
    #             number of cuts
    #     """
    #     poly = self.dataset

    #     planes = vtk.vtkPlanes()
    #     planes.SetOrigin(numpy2vtk(origins))
    #     planes.SetNormals(numpy2vtk(normals))

    #     cutter = vtk.vtkCutter()
    #     cutter.SetCutFunction(planes)
    #     cutter.SetInputData(poly)
    #     cutter.SetValue(0, 0.0)
    #     cutter.Update()

    #     msh = Mesh(cutter.GetOutput())
    #     msh.property.LightingOff()
    #     msh.property.SetColor(get_color("k2"))

    #     msh.pipeline = OperationNode(
    #         "intersect_with_multiplanes",
    #         parents=[self],
    #         comment=f"#pts {msh.dataset.GetNumberOfPoints()}",
    #     )
    #     return msh

    def collide_with(self, mesh2, tol=0, return_bool=False):
        """
        Collide this Mesh with the input surface.
        Information is stored in `ContactCells1` and `ContactCells2`.
        """
        ipdf = vtk.vtkCollisionDetectionFilter()
        # ipdf.SetGlobalWarningDisplay(0)

        transform0 = vtk.vtkTransform()
        transform1 = vtk.vtkTransform()

        # ipdf.SetBoxTolerance(tol)
        ipdf.SetCellTolerance(tol)
        ipdf.SetInputData(0, self.dataset)
        ipdf.SetInputData(1, mesh2)
        ipdf.SetTransform(0, transform0)
        ipdf.SetTransform(1, transform1)
        if return_bool:
            ipdf.SetCollisionModeToFirstContact()
        else:
            ipdf.SetCollisionModeToAllContacts()
        ipdf.Update()

        if return_bool:
            return bool(ipdf.GetNumberOfContacts())

        msh = Mesh(ipdf.GetContactsOutput(), "k", 1).lighting("off")
        msh.metadata["ContactCells1"] = vtk2numpy(
            ipdf.GetOutput(0).GetFieldData().GetArray("ContactCells")
        )
        msh.metadata["ContactCells2"] = vtk2numpy(
            ipdf.GetOutput(1).GetFieldData().GetArray("ContactCells")
        )
        msh.GetProperty().SetLineWidth(3)
        msh.name = "SurfaceCollision"

        msh.pipeline = OperationNode(
            "collide_with", parents=[self, mesh2],
            comment=f"#pts {msh.dataset.GetNumberOfPoints()}"
        )
        return msh

    def geodesic(self, start, end):
        """
        Dijkstra algorithm to compute the geodesic line.
        Takes as input a polygonal mesh and performs a single source shortest path calculation.

        Arguments:
            start : (int, list)
                start vertex index or close point `[x,y,z]`
            end :  (int, list)
                end vertex index or close point `[x,y,z]`

        Examples:
            - [geodesic.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/geodesic.py)

                ![](https://vedo.embl.es/images/advanced/geodesic.png)
        """
        if is_sequence(start):
            cc = self.vertices
            pa = Points(cc)
            start = pa.closest_point(start, return_point_id=True)
            end = pa.closest_point(end, return_point_id=True)

        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(self.dataset)
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
        dmesh.actor.SetProperty(prop)
        dmesh.property = prop
        dmesh.name = "GeodesicLine"

        dmesh.pipeline = OperationNode(
            "GeodesicLine", parents=[self], 
            comment=f"#pts {dmesh.dataset.GetNumberOfPoints()}"
        )
        return dmesh

    #####################################################################
    ### Stuff returning a Volume
    ###
    def binarize(
        self,
        spacing=(1, 1, 1),
        invert=False,
        direction_matrix=None,
        image_size=None,
        origin=None,
        fg_val=255,
        bg_val=0,
    ):
        """
        Convert a `Mesh` into a `Volume`
        where the foreground (exterior) voxels value is fg_val (255 by default)
        and the background (interior) voxels value is bg_val (0 by default).

        Examples:
            - [mesh2volume.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/mesh2volume.py)

                ![](https://vedo.embl.es/images/volumetric/mesh2volume.png)
        """
        # https://vtk.org/Wiki/VTK/Examples/Cxx/PolyData/PolyDataToImageData
        pd = self.dataset

        whiteImage = vtk.vtkImageData()
        if direction_matrix:
            whiteImage.SetDirectionMatrix(direction_matrix)

        dim = [0, 0, 0] if not image_size else image_size

        bounds = self.bounds()
        if not image_size:  # compute dimensions
            dim = [0, 0, 0]
            for i in [0, 1, 2]:
                dim[i] = int(np.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i]))

        whiteImage.SetDimensions(dim)
        whiteImage.SetSpacing(spacing)
        whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)

        if not origin:
            origin = [0, 0, 0]
            origin[0] = bounds[0] + spacing[0] / 2
            origin[1] = bounds[2] + spacing[1] / 2
            origin[2] = bounds[4] + spacing[2] / 2
        whiteImage.SetOrigin(origin)
        whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # fill the image with foreground voxels:
        inval = bg_val if invert else fg_val
        whiteImage.GetPointData().GetScalars().Fill(inval)

        # polygonal data --> image stencil:
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(pd)
        pol2stenc.SetOutputOrigin(whiteImage.GetOrigin())
        pol2stenc.SetOutputSpacing(whiteImage.GetSpacing())
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        # cut the corresponding white image and set the background:
        outval = fg_val if invert else bg_val

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(whiteImage)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.SetReverseStencil(invert)
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()
        vol = vedo.Volume(imgstenc.GetOutput())

        vol.pipeline = OperationNode(
            "binarize",
            parents=[self],
            comment=f"dim = {tuple(vol.dimensions())}",
            c="#e9c46a:#0096c7",
        )
        return vol

    def signed_distance(self, bounds=None, dims=(20, 20, 20), invert=False, maxradius=None):
        """
        Compute the `Volume` object whose voxels contains the signed distance from
        the mesh.

        Arguments:
            bounds : (list)
                bounds of the output volume
            dims : (list)
                dimensions (nr. of voxels) of the output volume
            invert : (bool)
                flip the sign

        Examples:
            - [volumeFromMesh.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/volumeFromMesh.py)
        """
        if maxradius is not None:
            vedo.logger.warning(
                "in signedDistance(maxradius=...) is ignored. (Only valid for pointclouds)."
            )
        if bounds is None:
            bounds = self.bounds()
        sx = (bounds[1] - bounds[0]) / dims[0]
        sy = (bounds[3] - bounds[2]) / dims[1]
        sz = (bounds[5] - bounds[4]) / dims[2]

        img = vtk.vtkImageData()
        img.SetDimensions(dims)
        img.SetSpacing(sx, sy, sz)
        img.SetOrigin(bounds[0], bounds[2], bounds[4])
        img.AllocateScalars(vtk.VTK_FLOAT, 1)

        imp = vtk.vtkImplicitPolyDataDistance()
        imp.SetInput(self.dataset)
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

        vol.pipeline = OperationNode(
            "signed_distance",
            parents=[self],
            comment=f"dim = {tuple(vol.dimensions())}",
            c="#e9c46a:#0096c7",
        )
        return vol

    def tetralize(
        self, side=0.02, nmax=300_000, gap=None, subsample=False, uniform=True, seed=0, debug=False
    ):
        """
        Tetralize a closed polygonal mesh. Return a `TetMesh`.

        Arguments:
            side : (float)
                desired side of the single tetras as fraction of the bounding box diagonal.
                Typical values are in the range (0.01 - 0.03)
            nmax : (int)
                maximum random numbers to be sampled in the bounding box
            gap : (float)
                keep this minimum distance from the surface,
                if None an automatic choice is made.
            subsample : (bool)
                subsample input surface, the geometry might be affected
                (the number of original faces reduceed), but higher tet quality might be obtained.
            uniform : (bool)
                generate tets more uniformly packed in the interior of the mesh
            seed : (int)
                random number generator seed
            debug : (bool)
                show an intermediate plot with sampled points

        Examples:
            - [tetralize_surface.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/tetralize_surface.py)

                ![](https://vedo.embl.es/images/volumetric/tetralize_surface.jpg)
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
            pts = (np.random.rand(n, 3) - 0.5) * np.array([x1 - x0, y1 - y0, z1 - z0]) + disp

        normals = surf.celldata["Normals"]
        cc = surf.cell_centers
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
        tcenters = tmesh.cell_centers

        ids = surf.inside_points(tcenters, return_ids=True)
        ins = np.zeros(tmesh.ncells)
        ins[ids] = 1

        if debug:
            # vedo.pyplot.histogram(fillpts.pointdata["Distance"], xtitle=f"gap={gap}").show().close()
            edges = self.edges
            points = self.vertices
            elen = mag(points[edges][:, 0, :] - points[edges][:, 1, :])
            histo = vedo.pyplot.histogram(elen, xtitle="edge length", xlim=(0, 3 * side * d))
            print(".. edges min, max", elen.min(), elen.max())
            fillpts.cmap("bone")
            vedo.show(
                [
                    [
                        f"This is a debug plot.\n\nGenerated points: {n}\ngap: {gap}",
                        surf.wireframe().alpha(0.2),
                        vedo.addons.Axes(surf),
                        fillpts,
                        Points(subpts).c("r4").ps(3),
                    ],
                    [f"Edges mean length: {np.mean(elen)}\n\nPress q to continue", histo],
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
            print(f".. tetralize() completed, ntets = {tmesh.ncells}")

        tmesh.pipeline = OperationNode(
            "tetralize", parents=[self], comment=f"#tets = {tmesh.ncells}", c="#e9c46a:#9e2a2b"
        )
        return tmesh
