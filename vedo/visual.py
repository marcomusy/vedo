#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import vedo
from vedo import colors
from vedo import utils


__docformat__ = "google"

__doc__ = """
Base classes to manage visualization and apperance of objects and their properties"
"""

__all__ = [
    "CommonVisual",
    "PointsVisual",
    "VolumeVisual",
    "MeshVisual",
    "PictureVisual",
    "Actor2D",
]


###################################################
class CommonVisual:
    """Class to manage the visual aspects common to all objects."""

    def __init__(self):

        self.mapper = None
        self.properties = None
        self.actor = None
        self.scalarbar = None        
        

    @property
    def LUT(self):
        """Return the lookup table of the object as a numpy object."""
        _lut = self.mapper.GetLookupTable()
        values = []
        for i in range(_lut.GetTable().GetNumberOfTuples()):
            # print("LUT i =", i, "value =", _lut.GetTableValue(i))
            values.append(_lut.GetTableValue(i))
        return np.array(values)

    @LUT.setter
    def LUT(self, arr):
        """
        Set the lookup table of the object from a numpy object.
        Consider using `cmap()` or `build_lut()` instead as it allows
        to set the range of the LUT and to use a string name for the color map.
        """
        _newlut = vtk.vtkLookupTable()
        _newlut.SetNumberOfTableValues(len(arr))
        if len(arr[0]) == 3:
            arr = np.insert(arr, 3, 1, axis=1)
        for i, v in enumerate(arr):
            _newlut.SetTableValue(i, v)
        _newlut.SetRange(self.mapper.GetScalarRange())
        _newlut.Build()
        self.mapper.SetLookupTable(_newlut)
        self.mapper.ScalarVisibilityOn()

    def add_observer(self, event_name, func, priority=0):
        """Add a callback function that will be called when an event occurs."""
        event_name = utils.get_vtk_name_event(event_name)
        idd = self.actor.AddObserver(event_name, func, priority)
        return idd

    def show(self, **options):
        """
        Create on the fly an instance of class `Plotter` or use the last existing one to
        show one single object.

        This method is meant as a shortcut. If more than one object needs to be visualised
        please use the syntax `show(mesh1, mesh2, volume, ..., options)`.

        Returns the `Plotter` class instance.
        """
        return vedo.plotter.show(self, **options)

    def thumbnail(self, zoom=1.25, size=(200, 200), bg="white", azimuth=0, elevation=0, axes=False):
        """Build a thumbnail of the object and return it as an array."""
        # speed is about 20Hz for size=[200,200]
        ren = vtk.vtkRenderer()

        actor = self.actor
        if isinstance(self, vedo.UGrid):
            geo = vtk.vtkGeometryFilter()
            geo.SetInputData(self.dataset)
            geo.Update()
            actor = vedo.Mesh(geo.GetOutput()).cmap("rainbow").actor

        ren.AddActor(actor)
        if axes:
            axes = vedo.addons.Axes(self)
            ren.AddActor(axes.actor)
        ren.ResetCamera()
        cam = ren.GetActiveCamera()
        cam.Zoom(zoom)
        cam.Elevation(elevation)
        cam.Azimuth(azimuth)

        ren_win = vtk.vtkRenderWindow()
        ren_win.SetOffScreenRendering(True)
        ren_win.SetSize(size)
        ren.SetBackground(colors.get_color(bg))
        ren_win.AddRenderer(ren)
        ren_win.Render()

        nx, ny = ren_win.GetSize()
        arr = vtk.vtkUnsignedCharArray()
        ren_win.GetRGBACharPixelData(0, 0, nx - 1, ny - 1, 0, arr)
        narr = utils.vtk2numpy(arr).T[:3].T.reshape([ny, nx, 3])
        narr = np.ascontiguousarray(np.flip(narr, axis=0))

        ren.RemoveActor(actor)
        if axes:
            ren.RemoveActor(axes.actor)
        ren_win.Finalize()
        del ren_win
        return narr

    def pickable(self, value=None):
        """Set/get the pickability property of an object."""
        if value is None:
            return self.actor.GetPickable()
        self.actor.SetPickable(value)
        return self

    def use_bounds(self, value=True):
        """
        Instruct the current camera to either take into account or ignore
        the object bounds when resetting.
        """
        self.actor.SetUseBounds(value)
        return self

    def draggable(self, value=None):  # NOT FUNCTIONAL?
        """Set/get the draggability property of an object."""
        if value is None:
            return self.actor.GetDragable()
        self.actor.SetDragable(value)
        return self

    def on(self):
        """Switch on  object visibility. Object is not removed."""
        self.actor.VisibilityOn()
        try:
            self.scalarbar.actor.VisibilityOn()
        except AttributeError:
            pass
        try:
            self.trail.actor.VisibilityOn()
        except AttributeError:
            pass
        try:
            for sh in self.shadows:
                sh.actor.VisibilityOn()
        except AttributeError:
            pass
        return self

    def off(self):
        """Switch off object visibility. Object is not removed."""
        self.actor.VisibilityOff()
        try:
            self.scalarbar.actor.VisibilityOff()
        except AttributeError:
            pass
        try:
            self.trail.actor.VisibilityOff()
        except AttributeError:
            pass
        try:
            for sh in self.shadows:
                sh.actor.VisibilityOff()
        except AttributeError:
            pass
        return self

    def toggle(self):
        """Toggle object visibility on/off."""
        v = self.actor.GetVisibility()
        if v:
            self.off()
        else:
            self.on()
        return self

    def add_scalarbar(
        self,
        title="",
        pos=(0.8, 0.05),
        title_yoffset=15,
        font_size=12,
        size=(None, None),
        nlabels=None,
        c=None,
        horizontal=False,
        use_alpha=True,
        label_format=":6.3g",
    ):
        """
        Add a 2D scalar bar for the specified obj.

        Examples:
            - [mesh_coloring.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_coloring.py)
            - [scalarbars.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/scalarbars.py)
        """
        plt = vedo.plotter_instance

        if plt and plt.renderer:
            c = (0.9, 0.9, 0.9)
            if np.sum(plt.renderer.GetBackground()) > 1.5:
                c = (0.1, 0.1, 0.1)
            if isinstance(self.scalarbar, vtk.vtkActor):
                plt.renderer.RemoveActor(self.scalarbar)
            elif isinstance(self.scalarbar, vedo.Assembly):
                for a in self.scalarbar.unpack():
                    plt.renderer.RemoveActor(a)
        if c is None:
            c = "gray"

        sb = vedo.addons.ScalarBar(
            self,
            title,
            pos,
            title_yoffset,
            font_size,
            size,
            nlabels,
            c,
            horizontal,
            use_alpha,
            label_format,
        )
        self.scalarbar = sb
        return self

    def add_scalarbar3d(
        self,
        title="",
        pos=None,
        size=(0, 0),
        title_font="",
        title_xoffset=-1.2,
        title_yoffset=0.0,
        title_size=1.5,
        title_rotation=0.0,
        nlabels=9,
        label_font="",
        label_size=1,
        label_offset=0.375,
        label_rotation=0,
        label_format="",
        italic=0,
        c=None,
        draw_box=True,
        above_text=None,
        below_text=None,
        nan_text="NaN",
        categories=None,
    ):
        """
        Associate a 3D scalar bar to the object and add it to the scene.
        The new scalarbar object (Assembly) will be accessible as obj.scalarbar

        Arguments:
            size : (list)
                (thickness, length) of scalarbar
            title : (str)
                scalar bar title
            title_xoffset : (float)
                horizontal space btw title and color scalarbar
            title_yoffset : (float)
                vertical space offset
            title_size : (float)
                size of title wrt numeric labels
            title_rotation : (float)
                title rotation in degrees
            nlabels : (int)
                number of numeric labels
            label_font : (str)
                font type for labels
            label_size : (float)
                label scale factor
            label_offset : (float)
                space btw numeric labels and scale
            label_rotation : (float)
                label rotation in degrees
            label_format : (str)
                label format for floats and integers (e.g. `':.2f'`)
            draw_box : (bool)
                draw a box around the colorbar
            categories : (list)
                make a categorical scalarbar,
                the input list will have the format `[value, color, alpha, textlabel]`

        Examples:
            - [scalarbars.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/scalarbars.py)
        """
        plt = vedo.plotter_instance
        if plt and c is None:  # automatic black or white
            c = (0.9, 0.9, 0.9)
            if np.sum(vedo.get_color(plt.backgrcol)) > 1.5:
                c = (0.1, 0.1, 0.1)
        if c is None:
            c = (0, 0, 0)
        c = vedo.get_color(c)

        self.scalarbar = vedo.addons.ScalarBar3D(
            self,
            title,
            pos,
            size,
            title_font,
            title_xoffset,
            title_yoffset,
            title_size,
            title_rotation,
            nlabels,
            label_font,
            label_size,
            label_offset,
            label_rotation,
            label_format,
            italic,
            c,
            draw_box,
            above_text,
            below_text,
            nan_text,
            categories,
        )
        return self

    def color(self, col, alpha=None, vmin=None, vmax=None):
        """
        Assign a color or a set of colors along the range of the scalar value.
        A single constant color can also be assigned.
        Any matplotlib color map name is also accepted, e.g. `volume.color('jet')`.

        E.g.: say that your cells scalar runs from -3 to 6,
        and you want -3 to show red and 1.5 violet and 6 green, then just set:

        `volume.color(['red', 'violet', 'green'])`

        You can also assign a specific color to a aspecific value with eg.:

        `volume.color([(0,'red'), (0.5,'violet'), (1,'green')])`

        Arguments:
            alpha : (list)
                use a list to specify transparencies along the scalar range
            vmin : (float)
                force the min of the scalar range to be this value
            vmax : (float)
                force the max of the scalar range to be this value
        """
        # supersedes method in Points, Mesh

        if col is None:
            return self

        if vmin is None:
            vmin, _ = self.dataset.GetScalarRange()
        if vmax is None:
            _, vmax = self.dataset.GetScalarRange()
        ctf = self.properties.GetRGBTransferFunction()
        ctf.RemoveAllPoints()

        if utils.is_sequence(col):
            if utils.is_sequence(col[0]) and len(col[0]) == 2:
                # user passing [(value1, color1), ...]
                for x, ci in col:
                    r, g, b = colors.get_color(ci)
                    ctf.AddRGBPoint(x, r, g, b)
                    # colors.printc('color at', round(x, 1),
                    #               'set to', colors.get_color_name((r, g, b)), bold=0)
            else:
                # user passing [color1, color2, ..]
                for i, ci in enumerate(col):
                    r, g, b = colors.get_color(ci)
                    x = vmin + (vmax - vmin) * i / (len(col) - 1)
                    ctf.AddRGBPoint(x, r, g, b)
        elif isinstance(col, str):
            if col in colors.colors.keys() or col in colors.color_nicks.keys():
                r, g, b = colors.get_color(col)
                ctf.AddRGBPoint(vmin, r, g, b)  # constant color
                ctf.AddRGBPoint(vmax, r, g, b)
            else:  # assume it's a colormap
                for x in np.linspace(vmin, vmax, num=64, endpoint=True):
                    r, g, b = colors.color_map(x, name=col, vmin=vmin, vmax=vmax)
                    ctf.AddRGBPoint(x, r, g, b)
        elif isinstance(col, int):
            r, g, b = colors.get_color(col)
            ctf.AddRGBPoint(vmin, r, g, b)  # constant color
            ctf.AddRGBPoint(vmax, r, g, b)
        else:
            vedo.logger.warning(f"in color() unknown input type {type(col)}")

        if alpha is not None:
            self.alpha(alpha, vmin=vmin, vmax=vmax)
        return self

    def alpha(self, alpha, vmin=None, vmax=None):
        """
        Assign a set of tranparencies along the range of the scalar value.
        A single constant value can also be assigned.

        E.g.: say `alpha=(0.0, 0.3, 0.9, 1)` and the scalar range goes from -10 to 150.
        Then all cells with a value close to -10 will be completely transparent, cells at 1/4
        of the range will get an alpha equal to 0.3 and voxels with value close to 150
        will be completely opaque.

        As a second option one can set explicit (x, alpha_x) pairs to define the transfer function.

        E.g.: say `alpha=[(-5, 0), (35, 0.4) (123,0.9)]` and the scalar range goes from -10 to 150.
        Then all cells below -5 will be completely transparent, cells with a scalar value of 35
        will get an opacity of 40% and above 123 alpha is set to 90%.
        """
        if vmin is None:
            vmin, _ = self.dataset.GetScalarRange()
        if vmax is None:
            _, vmax = self.dataset.GetScalarRange()
        otf = self.properties.GetScalarOpacity()
        otf.RemoveAllPoints()

        if utils.is_sequence(alpha):
            alpha = np.array(alpha)
            if len(alpha.shape) == 1:  # user passing a flat list e.g. (0.0, 0.3, 0.9, 1)
                for i, al in enumerate(alpha):
                    xalpha = vmin + (vmax - vmin) * i / (len(alpha) - 1)
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
                    # print("alpha at", round(xalpha, 1), "\tset to", al)
            elif len(alpha.shape) == 2:  # user passing [(x0,alpha0), ...]
                otf.AddPoint(vmin, alpha[0][1])
                for xalpha, al in alpha:
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
                otf.AddPoint(vmax, alpha[-1][1])

        else:

            otf.AddPoint(vmin, alpha)  # constant alpha
            otf.AddPoint(vmax, alpha)

        return self


###################################################
class PointsVisual(CommonVisual):
    """Class to manage the visual aspects of a ``Points`` object."""

    def __init__(self):
        # print("init PointsVisual")
        super().__init__()

    def clone2d(
        self,
        pos=(0, 0),
        coordsys=4,
        scale=None,
        c=None,
        alpha=None,
        ps=2,
        lw=1,
        sendback=False,
        layer=0,
    ):
        """
        Copy a 3D Mesh into a static 2D image. Returns a `vtkActor2D`.

        Arguments:
            coordsys : (int)
                the coordinate system, options are
                - 0 = Displays
                - 1 = Normalized Display
                - 2 = Viewport (origin is the bottom-left corner of the window)
                - 3 = Normalized Viewport
                - 4 = View (origin is the center of the window)
                - 5 = World (anchor the 2d image to mesh)

            ps : (int)
                point size in pixel units

            lw : (int)
                line width in pixel units

            sendback : (bool)
                put it behind any other 3D object

        Examples:
            - [clone2d.py](https://github.com/marcomusy/vedo/tree/master/examples/other/clone2d.py)

                ![](https://vedo.embl.es/images/other/clone2d.png)
        """
        if scale is None:
            msiz = self.diagonal_size()
            if vedo.plotter_instance and vedo.plotter_instance.window:
                sz = vedo.plotter_instance.window.GetSize()
                dsiz = utils.mag(sz)
                scale = dsiz / msiz / 10
            else:
                scale = 350 / msiz

        cmsh = self.clone()
        poly = cmsh.pos(0, 0, 0).scale(scale).dataset

        mapper3d = self.mapper
        cm = mapper3d.GetColorMode()
        lut = mapper3d.GetLookupTable()
        sv = mapper3d.GetScalarVisibility()
        use_lut = mapper3d.GetUseLookupTableScalarRange()
        vrange = mapper3d.GetScalarRange()
        sm = mapper3d.GetScalarMode()

        mapper2d = vtk.vtkPolyDataMapper2D()
        mapper2d.ShallowCopy(mapper3d)
        mapper2d.SetInputData(poly)
        mapper2d.SetColorMode(cm)
        mapper2d.SetLookupTable(lut)
        mapper2d.SetScalarVisibility(sv)
        mapper2d.SetUseLookupTableScalarRange(use_lut)
        mapper2d.SetScalarRange(vrange)
        mapper2d.SetScalarMode(sm)

        act2d = vtk.vtkActor2D()
        act2d.SetMapper(mapper2d)
        act2d.SetLayerNumber(layer)
        csys = act2d.GetPositionCoordinate()
        csys.SetCoordinateSystem(coordsys)
        act2d.SetPosition(pos)
        if c is not None:
            c = colors.get_color(c)
            act2d.GetProperty().SetColor(c)
            mapper2d.SetScalarVisibility(False)
        else:
            act2d.GetProperty().SetColor(cmsh.color())
        if alpha is not None:
            act2d.GetProperty().SetOpacity(alpha)
        else:
            act2d.GetProperty().SetOpacity(cmsh.alpha())
        act2d.GetProperty().SetPointSize(ps)
        act2d.GetProperty().SetLineWidth(lw)
        act2d.GetProperty().SetDisplayLocationToForeground()
        if sendback:
            act2d.GetProperty().SetDisplayLocationToBackground()

        # print(csys.GetCoordinateSystemAsString())
        # print(act2d.GetHeight(), act2d.GetWidth(), act2d.GetLayerNumber())
        return act2d

    ##################################################
    def copy_properties_from(self, source, deep=True, actor_related=True):
        """
        Copy properties from another ``Points`` object.
        """
        pr = vtk.vtkProperty()
        try:
            sp = source.properties
            mp = source.mapper
            sa = source.actor
        except AttributeError:
            sp = source.GetProperty()
            mp = source.GetMapper()
            sa = source
            
        if deep:
            pr.DeepCopy(sp)
        else:
            pr.ShallowCopy(sp)
        self.actor.SetProperty(pr)
        self.properties = pr

        if self.actor.GetBackfaceProperty():
            bfpr = vtk.vtkProperty()
            bfpr.DeepCopy(sa.GetBackfaceProperty())
            self.actor.SetBackfaceProperty(bfpr)
            self.properties_backface = bfpr

        if not actor_related:
            return self

        # mapper related:
        self.mapper.SetScalarVisibility(mp.GetScalarVisibility())
        self.mapper.SetScalarMode(mp.GetScalarMode())
        self.mapper.SetScalarRange(mp.GetScalarRange())
        self.mapper.SetLookupTable(mp.GetLookupTable())
        self.mapper.SetColorMode(mp.GetColorMode())
        self.mapper.SetInterpolateScalarsBeforeMapping(
            mp.GetInterpolateScalarsBeforeMapping()
        )
        self.mapper.SetUseLookupTableScalarRange(
            mp.GetUseLookupTableScalarRange()
        )

        self.actor.SetPickable(sa.GetPickable())
        self.actor.SetDragable(sa.GetDragable())
        self.actor.SetTexture(sa.GetTexture())
        self.actor.SetVisibility(sa.GetVisibility())
        return self

    def color(self, c=False, alpha=None):
        """
        Set/get mesh's color.
        If None is passed as input, will use colors from active scalars.
        Same as `mesh.c()`.
        """
        if c is False:
            return np.array(self.properties.GetColor())
        if c is None:
            self.mapper.ScalarVisibilityOn()
            return self
        self.mapper.ScalarVisibilityOff()
        cc = colors.get_color(c)
        self.properties.SetColor(cc)
        if self.trail:
            self.trail.GetProperty().SetColor(cc)
        if alpha is not None:
            self.alpha(alpha)
        return self

    def c(self, color=False, alpha=None):
        """
        Shortcut for `color()`.
        If None is passed as input, will use colors from current active scalars.
        """
        return self.color(color, alpha)

    def alpha(self, opacity=None):
        """Set/get mesh's transparency. Same as `mesh.opacity()`."""
        if opacity is None:
            return self.properties.GetOpacity()

        self.properties.SetOpacity(opacity)
        bfp = self.actor.GetBackfaceProperty()
        if bfp:
            if opacity < 1:
                self.properties_backface = bfp
                self.actor.SetBackfaceProperty(None)
            else:
                self.actor.SetBackfaceProperty(self.properties_backface)
        return self

    def lut_color_at(self, value):
        """
        Return the color of the lookup table at value.
        """
        lut = self.mapper.GetLookupTable()
        if not lut:
            return None
        rgb = [0,0,0]
        lut.GetColor(value, rgb)
        alpha = lut.GetOpacity(value)
        return np.array(rgb + [alpha])

    def opacity(self, alpha=None):
        """Set/get mesh's transparency. Same as `mesh.alpha()`."""
        return self.alpha(alpha)

    def force_opaque(self, value=True):
        """ Force the Mesh, Line or point cloud to be treated as opaque"""
        ## force the opaque pass, fixes picking in vtk9
        # but causes other bad troubles with lines..
        self.actor.SetForceOpaque(value)
        return self

    def force_translucent(self, value=True):
        """ Force the Mesh, Line or point cloud to be treated as translucent"""
        self.actor.SetForceTranslucent(value)
        return self

    def point_size(self, value=None):
        """Set/get mesh's point size of vertices. Same as `mesh.ps()`"""
        if value is None:
            return self.properties.GetPointSize()
            # self.properties.SetRepresentationToSurface()
        else:
            self.properties.SetRepresentationToPoints()
            self.properties.SetPointSize(value)
        return self

    def ps(self, pointsize=None):
        """Set/get mesh's point size of vertices. Same as `mesh.point_size()`"""
        return self.point_size(pointsize)

    def render_points_as_spheres(self, value=True):
        """Make points look spheric or else make them look as squares."""
        self.properties.SetRenderPointsAsSpheres(value)
        return self

    def lighting(
        self,
        style="",
        ambient=None,
        diffuse=None,
        specular=None,
        specular_power=None,
        specular_color=None,
        metallicity=None,
        roughness=None,
    ):
        """
        Set the ambient, diffuse, specular and specular_power lighting constants.

        Arguments:
            style : (str)
                preset style, options are `[metallic, plastic, shiny, glossy, ambient, off]`
            ambient : (float)
                ambient fraction of emission [0-1]
            diffuse : (float)
                emission of diffused light in fraction [0-1]
            specular : (float)
                fraction of reflected light [0-1]
            specular_power : (float)
                precision of reflection [1-100]
            specular_color : (color)
                color that is being reflected by the surface

        <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Phong_components_version_4.png" alt="", width=700px>

        Examples:
            - [specular.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/specular.py)
        """
        pr = self.properties

        if style:

            if style != "off":
                pr.LightingOn()

            if style == "off":
                pr.SetInterpolationToFlat()
                pr.LightingOff()
                return self  ##############

            if hasattr(pr, "GetColor"):  # could be Volume
                c = pr.GetColor()
            else:
                c = (1, 1, 0.99)
            mpr = self.mapper
            if hasattr(mpr, 'GetScalarVisibility') and mpr.GetScalarVisibility():
                c = (1,1,0.99)
            if   style=='metallic': pars = [0.1, 0.3, 1.0, 10, c]
            elif style=='plastic' : pars = [0.3, 0.4, 0.3,  5, c]
            elif style=='shiny'   : pars = [0.2, 0.6, 0.8, 50, c]
            elif style=='glossy'  : pars = [0.1, 0.7, 0.9, 90, (1,1,0.99)]
            elif style=='ambient' : pars = [0.8, 0.1, 0.0,  1, (1,1,1)]
            elif style=='default' : pars = [0.1, 1.0, 0.05, 5, c]
            else:
                vedo.logger.error("in lighting(): Available styles are")
                vedo.logger.error("[default, metallic, plastic, shiny, glossy, ambient, off]")
                raise RuntimeError()
            pr.SetAmbient(pars[0])
            pr.SetDiffuse(pars[1])
            pr.SetSpecular(pars[2])
            pr.SetSpecularPower(pars[3])
            if hasattr(pr, "GetColor"):
                pr.SetSpecularColor(pars[4])

        if ambient is not None: pr.SetAmbient(ambient)
        if diffuse is not None: pr.SetDiffuse(diffuse)
        if specular is not None: pr.SetSpecular(specular)
        if specular_power is not None: pr.SetSpecularPower(specular_power)
        if specular_color is not None: pr.SetSpecularColor(colors.get_color(specular_color))
        if utils.vtk_version_at_least(9):
            if metallicity is not None:
                pr.SetInterpolationToPBR()
                pr.SetMetallic(metallicity)
            if roughness is not None:
                pr.SetInterpolationToPBR()
                pr.SetRoughness(roughness)

        return self

    def point_blurring(self, r=1, emissive=False):
        """Set point blurring.
        Apply a gaussian convolution filter to the points.
        In this case the radius `r` is in absolute units of the mesh coordinates.
        With emissive set, the halo of point becomes light-emissive.
        """
        self.properties.SetRepresentationToPoints()
        if emissive:
            self.mapper.SetEmissive(bool(emissive))
        self.mapper.SetScaleFactor(r * 1.4142)

        # https://kitware.github.io/vtk-examples/site/Python/Meshes/PointInterpolator/
        if alpha < 1:
            self.mapper.SetSplatShaderCode(
                "//VTK::Color::Impl\n"
                "float dist = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);\n"
                "if (dist > 1.0) {\n"
                "   discard;\n"
                "} else {\n"
                f"  float scale = ({alpha} - dist);\n"
                "   ambientColor *= scale;\n"
                "   diffuseColor *= scale;\n"
                "}\n"
            )
            alpha = 1

        self.mapper.Modified()
        self.actor.Modified()
        self.properties.SetOpacity(alpha)
        self.actor.SetMapper(self.mapper)
        return self

    @property
    def cellcolors(self):
        """
        Colorize each cell (face) of a mesh by passing
        a 1-to-1 list of colors in format [R,G,B] or [R,G,B,A].
        Colors levels and opacities must be in the range [0,255].

        A single constant color can also be passed as string or RGBA.

        A cell array named "CellsRGBA" is automatically created.

        Examples:
            - [color_mesh_cells1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/color_mesh_cells1.py)
            - [color_mesh_cells2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/color_mesh_cells2.py)

            ![](https://vedo.embl.es/images/basic/colorMeshCells.png)
        """
        if "CellsRGBA" not in self.celldata.keys():
            lut = self.mapper.GetLookupTable()
            vscalars = self.dataset.GetCellData().GetScalars()
            if vscalars is None or lut is None:
                arr = np.zeros([self.ncells, 4], dtype=np.uint8)
                col = np.array(self.properties.GetColor())
                col = np.round(col * 255).astype(np.uint8)
                alf = self.properties.GetOpacity()
                alf = np.round(alf * 255).astype(np.uint8)
                arr[:, (0, 1, 2)] = col
                arr[:, 3] = alf
            else:
                cols = lut.MapScalars(vscalars, 0, 0)
                arr = utils.vtk2numpy(cols)
            self.celldata["CellsRGBA"] = arr
        self.celldata.select("CellsRGBA")
        return self.celldata["CellsRGBA"]

    @cellcolors.setter
    def cellcolors(self, value):
        if isinstance(value, str):
            c = colors.get_color(value)
            value = np.array([*c, 1]) * 255
            value = np.round(value)

        value = np.asarray(value)
        n = self.ncells

        if value.ndim == 1:
            value = np.repeat([value], n, axis=0)

        if value.shape[1] == 3:
            z = np.zeros((n, 1), dtype=np.uint8)
            value = np.append(value, z + 255, axis=1)

        assert n == value.shape[0]

        self.celldata["CellsRGBA"] = value.astype(np.uint8)
        # self.mapper.SetColorModeToDirectScalars() # done in select()
        self.celldata.select("CellsRGBA")

    @property
    def pointcolors(self):
        """
        Colorize each point (or vertex of a mesh) by passing
        a 1-to-1 list of colors in format [R,G,B] or [R,G,B,A].
        Colors levels and opacities must be in the range [0,255].

        A single constant color can also be passed as string or RGBA.

        A point array named "PointsRGBA" is automatically created.
        """
        if "PointsRGBA" not in self.pointdata.keys():
            lut = self.mapper.GetLookupTable()
            vscalars = self.dataset.GetPointData().GetScalars()
            if vscalars is None or lut is None:
                # create a constant array
                arr = np.zeros([self.npoints, 4], dtype=np.uint8)
                col = np.array(self.properties.GetColor())
                col = np.round(col * 255).astype(np.uint8)
                alf = self.properties.GetOpacity()
                alf = np.round(alf * 255).astype(np.uint8)
                arr[:, (0, 1, 2)] = col
                arr[:, 3] = alf
            else:
                cols = lut.MapScalars(vscalars, 0, 0)
                arr = utils.vtk2numpy(cols)
            self.pointdata["PointsRGBA"] = arr
        self.pointdata.select("PointsRGBA")
        return self.pointdata["PointsRGBA"]

    @pointcolors.setter
    def pointcolors(self, value):
        if isinstance(value, str):
            c = colors.get_color(value)
            value = np.array([*c, 1]) * 255
            value = np.round(value)

        value = np.asarray(value)
        n = self.npoints

        if value.ndim == 1:
            value = np.repeat([value], n, axis=0)

        if value.shape[1] == 3:
            z = np.zeros((n, 1), dtype=np.uint8)
            value = np.append(value, z + 255, axis=1)

        assert n == value.shape[0]

        self.pointdata["PointsRGBA"] = value.astype(np.uint8)
        # self.mapper.SetColorModeToDirectScalars() # done in select()
        self.pointdata.select("PointsRGBA")

    #####################################################################################
    def cmap(
        self,
        input_cmap,
        input_array=None,
        on="points",
        name="Scalars",
        vmin=None,
        vmax=None,
        n_colors=256,
        alpha=1.0,
        logscale=False,
    ):
        """
        Set individual point/cell colors by providing a list of scalar values and a color map.

        Arguments:
            input_cmap : (str, list, vtkLookupTable, matplotlib.colors.LinearSegmentedColormap)
                color map scheme to transform a real number into a color.
            input_array : (str, list, vtkArray)
                can be the string name of an existing array, a numpy array or a `vtkArray`.
            on : (str)
                either 'points' or 'cells'.
                Apply the color map to data which is defined on either points or cells.
            name : (str)
                give a name to the provided numpy array (if input_array is a numpy array)
            vmin : (float)
                clip scalars to this minimum value
            vmax : (float)
                clip scalars to this maximum value
            n_colors : (int)
                number of distinct colors to be used in colormap table.
            alpha : (float, list)
                Mesh transparency. Can be a `list` of values one for each vertex.
            logscale : (bool)
                Use logscale

        Examples:
            - [mesh_coloring.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_coloring.py)
            - [mesh_alphas.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_alphas.py)
            - [mesh_custom.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_custom.py)
            (and many others)

                ![](https://vedo.embl.es/images/basic/mesh_custom.png)
        """
        self._cmap_name = input_cmap

        if input_array is None:
            if not self.pointdata.keys() and self.celldata.keys():
                on = "cells"
                if not self.dataset.GetCellData().GetScalars():
                    input_array = 0  # pick the first at hand

        if "point" in on.lower():
            data = self.dataset.GetPointData()
            n = self.dataset.GetNumberOfPoints()
        elif "cell" in on.lower():
            data = self.dataset.GetCellData()
            n = self.dataset.GetNumberOfCells()
        else:
            vedo.logger.error("Must specify in cmap(on=...) to either 'cells' or 'points'")
            raise RuntimeError()

        if input_array is None:  # if None try to fetch the active scalars
            arr = data.GetScalars()
            if not arr:
                vedo.logger.error(f"in cmap(), cannot find any {on} active array ...skip coloring.")
                return self

            if not arr.GetName():  # sometimes arrays dont have a name..
                arr.SetName(name)

        elif isinstance(input_array, str):  # if a string is passed
            arr = data.GetArray(input_array)
            if not arr:
                vedo.logger.error(f"in cmap(), cannot find {on} array {input_array} ...skip coloring.")
                return self

        elif isinstance(input_array, int):  # if an int is passed
            if input_array < data.GetNumberOfArrays():
                arr = data.GetArray(input_array)
            else:
                vedo.logger.error(f"in cmap(), cannot find {on} array at {input_array} ...skip coloring.")
                return self

        elif utils.is_sequence(input_array):  # if a numpy array is passed
            npts = len(input_array)
            if npts != n:
                vedo.logger.error(f"in cmap(), nr. of input {on} scalars {npts} != {n} ...skip coloring.")
                return self
            arr = utils.numpy2vtk(input_array, name=name, dtype=float)
            data.AddArray(arr)
            data.Modified()

        elif isinstance(input_array, vtk.vtkArray):  # if a vtkArray is passed
            arr = input_array
            data.AddArray(arr)
            data.Modified()

        else:
            vedo.logger.error(f"in cmap(), cannot understand input type {type(input_array)}")
            raise RuntimeError()

        # Now we have array "arr"
        array_name = arr.GetName()

        if arr.GetNumberOfComponents() == 1:
            if vmin is None:
                vmin = arr.GetRange()[0]
            if vmax is None:
                vmax = arr.GetRange()[1]
        else:
            if vmin is None or vmax is None:
                vn = utils.mag(utils.vtk2numpy(arr))
            if vmin is None:
                vmin = vn.min()
            if vmax is None:
                vmax = vn.max()

        # interpolate alphas if they are not constant
        if not utils.is_sequence(alpha):
            alpha = [alpha] * n_colors
        else:
            v = np.linspace(0, 1, n_colors, endpoint=True)
            xp = np.linspace(0, 1, len(alpha), endpoint=True)
            alpha = np.interp(v, xp, alpha)

        ########################### build the look-up table
        if isinstance(input_cmap, vtk.vtkLookupTable):  # vtkLookupTable
            lut = input_cmap

        elif utils.is_sequence(input_cmap):  # manual sequence of colors
            lut = vtk.vtkLookupTable()
            if logscale:
                lut.SetScaleToLog10()
            lut.SetRange(vmin, vmax)
            ncols = len(input_cmap)
            lut.SetNumberOfTableValues(ncols)

            for i, c in enumerate(input_cmap):
                r, g, b = colors.get_color(c)
                lut.SetTableValue(i, r, g, b, alpha[i])
            lut.Build()

        else:  
            # assume string cmap name OR matplotlib.colors.LinearSegmentedColormap
            lut = vtk.vtkLookupTable()
            if logscale:
                lut.SetScaleToLog10()
            lut.SetVectorModeToMagnitude()
            lut.SetRange(vmin, vmax)
            lut.SetNumberOfTableValues(n_colors)
            mycols = colors.color_map(range(n_colors), input_cmap, 0, n_colors)
            for i, c in enumerate(mycols):
                r, g, b = c
                lut.SetTableValue(i, r, g, b, alpha[i])
            lut.Build()

        # TEST NEW WAY
        self.mapper.SetLookupTable(lut)
        self.mapper.ScalarVisibilityOn()
        self.mapper.SetColorModeToMapScalars()
        self.mapper.SetScalarRange(lut.GetRange())
        if "point" in on.lower():
            self.pointdata.select(array_name)
        else:
            self.celldata.select(array_name)
        return self

        # # TEST this is the old way:
        # # arr.SetLookupTable(lut) # wrong! causes weird instabilities with LUT
        # # if data.GetScalars():
        # #     data.GetScalars().SetLookupTable(lut)
        # #     data.GetScalars().Modified()

        # data.SetActiveScalars(array_name)
        # # data.SetScalars(arr)  # wrong! it deletes array in position 0, never use SetScalars
        # # data.SetActiveAttribute(array_name, 0) # boh!

        # self.mapper.SetLookupTable(lut)
        # self.mapper.SetColorModeToMapScalars()  # so we dont need to convert uint8 scalars

        # self.mapper.ScalarVisibilityOn()
        # self.mapper.SetScalarRange(lut.GetRange())

        # if on.startswith("point"):
        #     self.mapper.SetScalarModeToUsePointData()
        # else:
        #     self.mapper.SetScalarModeToUseCellData()
        # if hasattr(self.mapper, "SetArrayName"):
        #     self.mapper.SetArrayName(array_name)
        # return self

    def add_trail(self, offset=(0, 0, 0), n=50, c=None, alpha=1.0, lw=2):
        """
        Add a trailing line to mesh.
        This new mesh is accessible through `mesh.trail`.

        Arguments:
            offset : (float)
                set an offset vector from the object center.
            n : (int)
                number of segments
            lw : (float)
                line width of the trail

        Examples:
            - [trail.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/trail.py)

                ![](https://vedo.embl.es/images/simulations/trail.gif)

            - [airplane1.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplane1.py)
            - [airplane2.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplane2.py)
        """
        if self.trail is None:
            pos = self.pos()
            self.trail_offset = np.asarray(offset)
            self.trail_points = [pos] * n

            if c is None:
                col = self.properties.GetColor()
            else:
                col = colors.get_color(c)

            tline = vedo.shapes.Line(pos, pos, res=n, c=col, alpha=alpha, lw=lw)
            self.trail = tline  # holds the Line
        return self

    def update_trail(self):
        """
        Update the trailing line of a moving object.
        """
        currentpos = self.pos()
        self.trail_points.append(currentpos)  # cycle
        self.trail_points.pop(0)
        data = np.array(self.trail_points) + self.trail_offset
        tpoly = self.trail.dataset
        tpoly.GetPoints().SetData(utils.numpy2vtk(data, dtype=np.float32))
        return self

    def _compute_shadow(self, plane, point, direction):
        shad = self.clone()
        shad.dataset.GetPointData().SetTCoords(None)  # remove any texture coords
        shad.name = "Shadow"

        pts = shad.vertices
        if plane == "x":
            # shad = shad.project_on_plane('x')
            # instead do it manually so in case of alpha<1
            # we dont see glitches due to coplanar points
            # we leave a small tolerance of 0.1% in thickness
            x0, x1 = self.xbounds()
            pts[:, 0] = (pts[:, 0] - (x0 + x1) / 2) / 1000 + self.actor.GetOrigin()[0]
            shad.vertices = pts
            shad.x(point)
        elif plane == "y":
            x0, x1 = self.ybounds()
            pts[:, 1] = (pts[:, 1] - (x0 + x1) / 2) / 1000 + self.actor.GetOrigin()[1]
            shad.vertices = pts
            shad.y(point)
        elif plane == "z":
            x0, x1 = self.zbounds()
            pts[:, 2] = (pts[:, 2] - (x0 + x1) / 2) / 1000 + self.actor.GetOrigin()[2]
            shad.vertices = pts
            shad.z(point)
        else:
            shad = shad.project_on_plane(plane, point, direction)
        return shad

    def add_shadow(self, plane, point, direction=None, c=(0.6, 0.6, 0.6), alpha=1, culling=0):
        """
        Generate a shadow out of an `Mesh` on one of the three Cartesian planes.
        The output is a new `Mesh` representing the shadow.
        This new mesh is accessible through `mesh.shadow`.
        By default the shadow mesh is placed on the bottom wall of the bounding box.

        See also `pointcloud.project_on_plane()`.

        Arguments:
            plane : (str, Plane)
                if plane is `str`, plane can be one of `['x', 'y', 'z']`,
                represents x-plane, y-plane and z-plane, respectively.
                Otherwise, plane should be an instance of `vedo.shapes.Plane`
            point : (float, array)
                if plane is `str`, point should be a float represents the intercept.
                Otherwise, point is the camera point of perspective projection
            direction : (list)
                direction of oblique projection
            culling : (int)
                choose between front [1] or backface [-1] culling or None.

        Examples:
            - [shadow1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/shadow1.py)
            - [airplane1.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplane1.py)
            - [airplane2.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplane2.py)

            ![](https://vedo.embl.es/images/simulations/57341963-b8910900-713c-11e9-898a-84b6d3712bce.gif)
        """
        shad = self._compute_shadow(plane, point, direction)
        shad.c(c).alpha(alpha)

        try:
            # Points dont have these methods
            shad.flat()
            if culling in (1, True):
                shad.frontface_culling()
            elif culling == -1:
                shad.backface_culling()
        except AttributeError:
            pass

        shad.properties.LightingOff()
        shad.actor.SetPickable(False)
        shad.actor.SetUseBounds(True)

        if shad not in self.shadows:
            self.shadows.append(shad)
            shad.info = dict(plane=plane, point=point, direction=direction)
        return self

    def update_shadows(self):
        """
        Update the shadows of a moving object.
        """
        for sha in self.shadows:
            plane = sha.info["plane"]
            point = sha.info["point"]
            direction = sha.info["direction"]
            new_sha = self._compute_shadow(plane, point, direction)
            # sha.DeepCopy(new_sha)
            sha._update(new_sha.dataset)
        return self

    def labels(
        self,
        content=None,
        on="points",
        scale=None,
        xrot=0.0,
        yrot=0.0,
        zrot=0.0,
        ratio=1,
        precision=None,
        italic=False,
        font="",
        justify="bottom-left",
        c="black",
        alpha=1.0,
        cells=None,
    ):
        """
        Generate value or ID labels for mesh cells or points.
        For large nr. of labels use `font="VTK"` which is much faster.

        See also:
            `labels2d()`, `flagpole()`, `caption()` and `legend()`.

        Arguments:
            content : (list,int,str)
                either 'id', 'cellid', array name or array number.
                A array can also be passed (must match the nr. of points or cells).
            on : (str)
                generate labels for "cells" instead of "points"
            scale : (float)
                absolute size of labels, if left as None it is automatic
            zrot : (float)
                local rotation angle of label in degrees
            ratio : (int)
                skipping ratio, to reduce nr of labels for large meshes
            precision : (int)
                numeric precision of labels

        ```python
        from vedo import *
        s = Sphere(res=10).linewidth(1).c("orange").compute_normals()
        point_ids = s.labels('id', on="points").c('green')
        cell_ids  = s.labels('id', on="cells" ).c('black')
        show(s, point_ids, cell_ids)
        ```
        ![](https://vedo.embl.es/images/feats/labels.png)

        Examples:
            - [boundaries.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/boundaries.py)

                ![](https://vedo.embl.es/images/basic/boundaries.png)
        """
        if cells is not None:  # deprecation message
            vedo.logger.warning("In labels(cells=...) please use labels(on='cells') instead")

        if "cell" in on or "face" in on:
            cells = True

        if isinstance(content, str):
            if content in ("cellid", "cellsid"):
                cells = True
                content = "id"

        if cells:
            elems = self.cell_centers
            # norms = self.normals(cells=True, recompute=False)
            norms = self.cell_normals
            ns = np.sqrt(self.ncells)
        else:
            elems = self.vertices
            # norms = self.normals(cells=False, recompute=False)
            norms = self.vertex_normals
            ns = np.sqrt(self.npoints)

        hasnorms = False
        if len(norms) > 0:
            hasnorms = True

        if scale is None:
            if not ns:
                ns = 100
            scale = self.diagonal_size() / ns / 10

        arr = None
        mode = 0
        if content is None:
            mode = 0
            if cells:
                if self.dataset.GetCellData().GetScalars():
                    name = self.dataset.GetCellData().GetScalars().GetName()
                    arr = self.celldata[name]
            else:
                if self.dataset.GetPointData().GetScalars():
                    name = self.dataset.GetPointData().GetScalars().GetName()
                    arr = self.pointdata[name]
        elif isinstance(content, (str, int)):
            if content == "id":
                mode = 1
            elif cells:
                mode = 0
                arr = self.celldata[content]
            else:
                mode = 0
                arr = self.pointdata[content]
        elif utils.is_sequence(content):
            mode = 0
            arr = content
            # print('WEIRD labels() test', content)
            # exit()

        if arr is None and mode == 0:
            vedo.logger.error("in labels(), array not found for points or cells")
            return None

        tapp = vtk.vtkAppendPolyData()
        ninputs = 0

        for i, e in enumerate(elems):
            if i % ratio:
                continue

            if mode == 1:
                txt_lab = str(i)
            else:
                if precision:
                    txt_lab = utils.precision(arr[i], precision)
                else:
                    txt_lab = str(arr[i])

            if not txt_lab:
                continue

            if font == "VTK":
                tx = vtk.vtkVectorText()
                tx.SetText(txt_lab)
                tx.Update()
                tx_poly = tx.GetOutput()
            else:
                tx_poly = vedo.shapes.Text3D(txt_lab, font=font, justify=justify).dataset

            if tx_poly.GetPointData() == 0:
                continue  #######################
            ninputs += 1

            T = vtk.vtkTransform()
            T.PostMultiply()
            if italic:
                T.Concatenate([1,0.2,0,0,
                               0,1,0,0,
                               0,0,1,0,
                               0,0,0,1])
            if hasnorms:
                ni = norms[i]
                if cells:  # center-justify
                    bb = tx_poly.GetBounds()
                    dx, dy = (bb[1] - bb[0]) / 2, (bb[3] - bb[2]) / 2
                    T.Translate(-dx, -dy, 0)
                if xrot:
                    T.RotateX(xrot)
                if yrot:
                    T.RotateY(yrot)
                if zrot:
                    T.RotateZ(zrot)
                crossvec = np.cross([0, 0, 1], ni)
                angle = np.arccos(np.dot([0, 0, 1], ni)) * 57.3
                T.RotateWXYZ(angle, crossvec)
                if cells:  # small offset along normal only for cells
                    T.Translate(ni * scale / 2)
            else:
                if xrot:
                    T.RotateX(xrot)
                if yrot:
                    T.RotateY(yrot)
                if zrot:
                    T.RotateZ(zrot)
            T.Scale(scale, scale, scale)
            T.Translate(e)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(tx_poly)
            tf.SetTransform(T)
            tf.Update()
            tapp.AddInputData(tf.GetOutput())

        if ninputs:
            tapp.Update()
            lpoly = tapp.GetOutput()
        else:  # return an empty obj
            lpoly = vtk.vtkPolyData()

        ids = vedo.mesh.Mesh(lpoly, c=c, alpha=alpha)
        ids.properties.LightingOff()
        ids.actor.PickableOff()
        ids.actor.SetUseBounds(False)
        return ids

    def labels2d(
        self,
        content="id",
        on="points",
        scale=1.0,
        precision=4,
        font="Calco",
        justify="bottom-left",
        angle=0.0,
        frame=False,
        c="black",
        bc=None,
        alpha=1.0,
    ):
        """
        Generate value or ID bi-dimensional labels for mesh cells or points.

        See also: `labels()`, `flagpole()`, `caption()` and `legend()`.

        Arguments:
            content : (str)
                either 'id', 'cellid', or array name
            on : (str)
                generate labels for "cells" instead of "points" (the default)
            scale : (float)
                size scaling of labels
            precision : (int)
                precision of numeric labels
            angle : (float)
                local rotation angle of label in degrees
            frame : (bool)
                draw a frame around the label
            bc : (str)
                background color of the label

        ```python
        from vedo import Sphere, show
        sph = Sphere(quads=True, res=4).compute_normals().wireframe()
        sph.celldata["zvals"] = sph.cell_centers[:,2]
        l2d = sph.labels("zvals", on="cells", precision=2).backcolor('orange9')
        show(sph, l2d, axes=1).close()
        ```
        ![](https://vedo.embl.es/images/feats/labels2d.png)
        """
        cells = False
        if isinstance(content, str):
            if content in ("cellid", "cellsid"):
                cells = True
                content = "id"

        if "cell" in on:
            cells = True
        elif "point" in on:
            cells = False

        if cells:
            if content != "id" and content not in self.celldata.keys():
                vedo.logger.error(f"In labels2d: cell array {content} does not exist.")
                return None
            cellcloud = vedo.Points(self.cell_centers)
            arr = self.dataset.GetCellData().GetScalars()
            poly = cellcloud.dataset
            poly.GetPointData().SetScalars(arr)
        else:
            poly = self.dataset
            if content != "id" and content not in self.pointdata.keys():
                vedo.logger.error(f"In labels2d: point array {content} does not exist.")
                return None
            self.pointdata.select(content)

        mp = vtk.vtkLabeledDataMapper()

        if content == "id":
            mp.SetLabelModeToLabelIds()
        else:
            mp.SetLabelModeToLabelScalars()
            if precision is not None:
                mp.SetLabelFormat(f"%-#.{precision}g")

        pr = mp.GetLabelTextProperty()
        c = colors.get_color(c)
        pr.SetColor(c)
        pr.SetOpacity(alpha)
        pr.SetFrame(frame)
        pr.SetFrameColor(c)
        pr.SetItalic(False)
        pr.BoldOff()
        pr.ShadowOff()
        pr.UseTightBoundingBoxOn()
        pr.SetOrientation(angle)
        pr.SetFontFamily(vtk.VTK_FONT_FILE)
        fl = utils.get_font_path(font)
        pr.SetFontFile(fl)
        pr.SetFontSize(int(20 * scale))

        if "cent" in justify or "mid" in justify:
            pr.SetJustificationToCentered()
        elif "rig" in justify:
            pr.SetJustificationToRight()
        elif "left" in justify:
            pr.SetJustificationToLeft()
        # ------
        if "top" in justify:
            pr.SetVerticalJustificationToTop()
        else:
            pr.SetVerticalJustificationToBottom()

        if bc is not None:
            bc = colors.get_color(bc)
            pr.SetBackgroundColor(bc)
            pr.SetBackgroundOpacity(alpha)

        mp.SetInputData(poly)
        a2d = vtk.vtkActor2D()
        a2d.PickableOff()
        a2d.SetMapper(mp)
        return a2d

    def legend(self, txt):
        """Book a legend text."""
        self.info["legend"] = txt
        return self

    def flagpole(
        self,
        txt=None,
        point=None,
        offset=None,
        s=None,
        font="Calco",
        rounded=True,
        c=None,
        alpha=1.0,
        lw=2,
        italic=0.0,
        padding=0.1,
    ):
        """
        Generate a flag pole style element to describe an object.
        Returns a `Mesh` object.

        Use flagpole.follow_camera() to make it face the camera in the scene.

        Consider using `settings.use_parallel_projection = True` 
        to avoid perspective distortions.

        See also `flagpost()`.

        Arguments:
            txt : (str)
                Text to display. The default is the filename or the object name.
            point : (list)
                position of the flagpole pointer. 
            offset : (list)
                text offset wrt the application point. 
            s : (float)
                size of the flagpole.
            font : (str)
                font face. Check [available fonts here](https://vedo.embl.es/fonts).
            rounded : (bool)
                draw a rounded or squared box around the text.
            c : (list)
                text and box color.
            alpha : (float)
                opacity of text and box.
            lw : (float)
                line with of box frame.
            italic : (float)
                italicness of text.

        Examples:
            - [intersect2d.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/intersect2d.py)

                ![](https://vedo.embl.es/images/pyplot/intersect2d.png)

            - [goniometer.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/goniometer.py)
            - [flag_labels1.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels1.py)
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels2.py)
        """
        objs = []

        if txt is None:
            if self.filename:
                txt = self.filename.split("/")[-1]
            elif self.name:
                txt = self.name
            else:
                return None

        x0, x1, y0, y1, z0, z1 = self.bounds()
        d = self.diagonal_size()
        if point is None:
            if d:
                point = self.closest_point([(x0 + x1) / 2, (y0 + y1) / 2, z1])
                # point = self.closest_point([x1, y0, z1])
            else:  # it's a Point
                point = self.transform.position

        pt = utils.make3d(point)

        if offset is None:
            offset = [(x1 - x0) / 1.75, (y1 - y0) / 5, 0]
        offset = utils.make3d(offset)

        if s is None:
            s = d / 20

        sph = None
        if d and (z1 - z0) / d > 0.1:
            sph = vedo.shapes.Sphere(pt, r=s * 0.4, res=6)

        if c is None:
            c = np.array(self.color()) / 1.4

        lab = vedo.shapes.Text3D(
            txt, pos=pt + offset, s=s, font=font, italic=italic, justify="center"
        )
        objs.append(lab)

        if d and not sph:
            sph = vedo.shapes.Circle(pt, r=s / 3, res=16)
        objs.append(sph)

        x0, x1, y0, y1, z0, z1 = lab.bounds()
        aline = [(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0)]
        if rounded:
            box = vedo.shapes.KSpline(aline, closed=True)
        else:
            box = vedo.shapes.Line(aline, closed=True)

        cnt = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]

        # box.actor.SetOrigin(cnt)
        box.scale([1 + padding, 1 + 2 * padding, 1], origin=cnt)
        objs.append(box)

        x0, x1, y0, y1, z0, z1 = box.bounds()
        if x0 < pt[0] < x1:
            c0 = box.closest_point(pt)
            c1 = [c0[0], c0[1] + (pt[1] - y0) / 4, pt[2]]
        elif (pt[0] - x0) < (x1 - pt[0]):
            c0 = [x0, (y0 + y1) / 2, pt[2]]
            c1 = [x0 + (pt[0] - x0) / 4, (y0 + y1) / 2, pt[2]]
        else:
            c0 = [x1, (y0 + y1) / 2, pt[2]]
            c1 = [x1 + (pt[0] - x1) / 4, (y0 + y1) / 2, pt[2]]

        con = vedo.shapes.Line([c0, c1, pt])
        objs.append(con)

        mobjs = vedo.merge(objs).c(c).alpha(alpha)
        mobjs.name = "FlagPole"
        mobjs.bc("tomato").pickable(False)
        mobjs.properties.LightingOff()
        mobjs.properties.SetLineWidth(lw)
        mobjs.actor.UseBoundsOff()
        mobjs.actor.SetPosition([0,0,0])
        mobjs.actor.SetOrigin(pt)
        # print(pt)
        return mobjs

        # mobjs = vedo.Assembly(objs)#.c(c).alpha(alpha)
        # mobjs.name = "FlagPole"
        # # mobjs.bc("tomato").pickable(False)
        # # mobjs.properties.LightingOff()
        # # mobjs.properties.SetLineWidth(lw)
        # # mobjs.actor.UseBoundsOff()
        # # mobjs.actor.SetPosition([0,0,0])
        # # mobjs.actor.SetOrigin(pt)
        # # print(pt)
        # return mobjs

    def flagpost(
        self,
        txt=None,
        point=None,
        offset=None,
        s=1.0,
        c="k9",
        bc="k1",
        alpha=1,
        lw=0,
        font="Calco",
        justify="center-left",
        vspacing=1.0,
    ):
        """
        Generate a flag post style element to describe an object.

        Arguments:
            txt : (str)
                Text to display. The default is the filename or the object name.
            point : (list)
                position of the flag anchor point. The default is None.
            offset : (list)
                a 3D displacement or offset. The default is None.
            s : (float)
                size of the text to be shown
            c : (list)
                color of text and line
            bc : (list)
                color of the flag background
            alpha : (float)
                opacity of text and box.
            lw : (int)
                line with of box frame. The default is 0.
            font : (str)
                font name. Use a monospace font for better rendering. The default is "Calco".
                Type `vedo -r fonts` for a font demo.
                Check [available fonts here](https://vedo.embl.es/fonts).
            justify : (str)
                internal text justification. The default is "center-left".
            vspacing : (float)
                vertical spacing between lines.

        Examples:
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/other/flag_labels2.py)

            ![](https://vedo.embl.es/images/other/flag_labels2.png)
        """
        if txt is None:
            if self.filename:
                txt = self.filename.split("/")[-1]
            elif self.name:
                txt = self.name
            else:
                return None

        x0, x1, y0, y1, z0, z1 = self.bounds()
        d = self.diagonal_size()
        if point is None:
            if d:
                point = self.closest_point([(x0 + x1) / 2, (y0 + y1) / 2, z1])
            else:  # it's a Point
                point = self.transform.position

        point = utils.make3d(point)

        if offset is None:
            offset = [0, 0, (z1 - z0) / 2]
        offset = utils.make3d(offset)

        fpost = vedo.addons.Flagpost(
            txt, point, point + offset, s, c, bc, alpha, lw, font, justify, vspacing
        )
        self._caption = fpost
        return fpost

    def caption(
        self,
        txt=None,
        point=None,
        size=(0.30, 0.15),
        padding=5,
        font="Calco",
        justify="center-right",
        vspacing=1.0,
        c=None,
        alpha=1.0,
        lw=1,
        ontop=True,
    ):
        """
        Create a 2D caption to an object which follows the camera movements.
        Latex is not supported. Returns the same input object for concatenation.

        See also `flagpole()`, `flagpost()`, `labels()` and `legend()`
        with similar functionality.

        Arguments:
            txt : (str)
                text to be rendered. The default is the file name.
            point : (list)
                anchoring point. The default is None.
            size : (list)
                (width, height) of the caption box. The default is (0.30, 0.15).
            padding : (float)
                padding space of the caption box in pixels. The default is 5.
            font : (str)
                font name. Use a monospace font for better rendering. The default is "VictorMono".
                Type `vedo -r fonts` for a font demo.
                Check [available fonts here](https://vedo.embl.es/fonts).
            justify : (str)
                internal text justification. The default is "center-right".
            vspacing : (float)
                vertical spacing between lines. The default is 1.
            c : (str)
                text and box color. The default is 'lb'.
            alpha : (float)
                text and box transparency. The default is 1.
            lw : (int)
                line width in pixels. The default is 1.
            ontop : (bool)
                keep the 2d caption always on top. The default is True.

        Examples:
            - [caption.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/caption.py)

                ![](https://vedo.embl.es/images/pyplot/caption.png)

            - [flag_labels1.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels1.py)
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels2.py)
        """
        if txt is None:
            if self.filename:
                txt = self.filename.split("/")[-1]
            elif self.name:
                txt = self.name

        if not txt:  # disable it
            self._caption = None
            return self

        for r in vedo.shapes._reps:
            txt = txt.replace(r[0], r[1])

        if c is None:
            c = np.array(self.properties.GetColor()) / 2
        else:
            c = colors.get_color(c)

        if point is None:
            x0, x1, y0, y1, _, z1 = self.dataset.GetBounds()
            pt = [(x0 + x1) / 2, (y0 + y1) / 2, z1]
            point = self.closest_point(pt)

        capt = vtk.vtkCaptionActor2D()
        capt.SetAttachmentPoint(point)
        capt.SetBorder(True)
        capt.SetLeader(True)
        sph = vtk.vtkSphereSource()
        sph.Update()
        capt.SetLeaderGlyphData(sph.GetOutput())
        capt.SetMaximumLeaderGlyphSize(5)
        capt.SetPadding(int(padding))
        capt.SetCaption(txt)
        capt.SetWidth(size[0])
        capt.SetHeight(size[1])
        capt.SetThreeDimensionalLeader(not ontop)

        pra = capt.GetProperty()
        pra.SetColor(c)
        pra.SetOpacity(alpha)
        pra.SetLineWidth(lw)

        pr = capt.GetCaptionTextProperty()
        pr.SetFontFamily(vtk.VTK_FONT_FILE)
        fl = utils.get_font_path(font)
        pr.SetFontFile(fl)
        pr.ShadowOff()
        pr.BoldOff()
        pr.FrameOff()
        pr.SetColor(c)
        pr.SetOpacity(alpha)
        pr.SetJustificationToLeft()
        if "top" in justify:
            pr.SetVerticalJustificationToTop()
        if "bottom" in justify:
            pr.SetVerticalJustificationToBottom()
        if "cent" in justify:
            pr.SetVerticalJustificationToCentered()
            pr.SetJustificationToCentered()
        if "left" in justify:
            pr.SetJustificationToLeft()
        if "right" in justify:
            pr.SetJustificationToRight()
        pr.SetLineSpacing(vspacing)
        return capt


#####################################################################
class MeshVisual(PointsVisual):
    """Class to manage the visual aspects of a ``Maesh`` object."""

    def __init__(self) -> None:
        # print("INIT MeshVisual", super())
        super().__init__()

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
        factor.SetProperty(self.properties)
        factor.SetBackfaceProperty(self.actor.GetBackfaceProperty())
        factor.SetTexture(self.actor.GetTexture())
        factor.SetScale(self.actor.GetScale())
        # factor.SetOrientation(self.actor.GetOrientation())
        factor.SetPosition(self.actor.GetPosition())
        factor.SetUseBounds(self.actor.GetUseBounds())

        if origin is None:
            factor.SetOrigin(self.actor.GetOrigin())
        else:
            factor.SetOrigin(origin)

        factor.PickableOff()

        if isinstance(camera, vtk.vtkCamera):
            factor.SetCamera(camera)
        else:
            plt = vedo.plotter_instance
            if plt and plt.renderer and plt.renderer.GetActiveCamera():
                factor.SetCamera(plt.renderer.GetActiveCamera())

        self.actor = None
        factor.data = self
        self.actor = factor
        return self

    def wireframe(self, value=True):
        """Set mesh's representation as wireframe or solid surface."""
        if value:
            self.properties.SetRepresentationToWireframe()
        else:
            self.properties.SetRepresentationToSurface()
        return self

    def flat(self):
        """Set surface interpolation to flat.

        <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Phong_components_version_4.png" width="700">
        """
        self.properties.SetInterpolationToFlat()
        return self

    def phong(self):
        """Set surface interpolation to "phong"."""
        self.properties.SetInterpolationToPhong()
        return self

    def backface_culling(self, value=True):
        """Set culling of polygons based on orientation of normal with respect to camera."""
        self.properties.SetBackfaceCulling(value)
        return self

    def render_lines_as_tubes(self, value=True):
        """Wrap a fake tube around a simple line for visualization"""
        self.properties.SetRenderLinesAsTubes(value)
        return self

    def frontface_culling(self, value=True):
        """Set culling of polygons based on orientation of normal with respect to camera."""
        self.properties.SetFrontfaceCulling(value)
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

        if self.properties.GetOpacity() < 1:
            return self

        if not back_prop:
            back_prop = vtk.vtkProperty()

        back_prop.SetDiffuseColor(colors.get_color(bc))
        back_prop.SetOpacity(self.properties.GetOpacity())
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
                self.properties.EdgeVisibilityOff()
                self.properties.SetRepresentationToSurface()
                return self
            self.properties.EdgeVisibilityOn()
            self.properties.SetLineWidth(lw)
        else:
            return self.properties.GetLineWidth()
        return self

    def lw(self, linewidth=None):
        """Set/get width of mesh edges. Same as `linewidth()`."""
        return self.linewidth(linewidth)

    def linecolor(self, lc=None):
        """Set/get color of mesh edges. Same as `lc()`."""
        if lc is None:
            return self.properties.GetEdgeColor()
        self.properties.EdgeVisibilityOn()
        self.properties.SetEdgeColor(colors.get_color(lc))
        return self

    def lc(self, linecolor=None):
        """Set/get color of mesh edges. Same as `linecolor()`."""
        return self.linecolor(linecolor)


########################################################################################
class VolumeVisual(CommonVisual):
    """Class to manage the visual aspects of a ``Volume`` object."""

    def __init__(self) -> None:
        # print("INIT VolumeVisual")
        super().__init__()

    def alpha_unit(self, u=None):
        """
        Defines light attenuation per unit length. Default is 1.
        The larger the unit length, the further light has to travel to attenuate the same amount.

        E.g., if you set the unit distance to 0, you will get full opacity.
        It means that when light travels 0 distance it's already attenuated a finite amount.
        Thus, any finite distance should attenuate all light.
        The larger you make the unit distance, the more transparent the rendering becomes.
        """
        if u is None:
            return self.properties.GetScalarOpacityUnitDistance()
        self.properties.SetScalarOpacityUnitDistance(u)
        return self

    def alpha_gradient(self, alpha_grad, vmin=None, vmax=None):
        """
        Assign a set of tranparencies to a volume's gradient
        along the range of the scalar value.
        A single constant value can also be assigned.
        The gradient function is used to decrease the opacity
        in the "flat" regions of the volume while maintaining the opacity
        at the boundaries between material types.  The gradient is measured
        as the amount by which the intensity changes over unit distance.

        The format for alpha_grad is the same as for method `volume.alpha()`.
        """
        if vmin is None:
            vmin, _ = self.dataset.GetScalarRange()
        if vmax is None:
            _, vmax = self.dataset.GetScalarRange()

        if alpha_grad is None:
            self.properties.DisableGradientOpacityOn()
            return self

        self.properties.DisableGradientOpacityOff()

        gotf = self.properties.GetGradientOpacity()
        if utils.is_sequence(alpha_grad):
            alpha_grad = np.array(alpha_grad)
            if len(alpha_grad.shape) == 1:  # user passing a flat list e.g. (0.0, 0.3, 0.9, 1)
                for i, al in enumerate(alpha_grad):
                    xalpha = vmin + (vmax - vmin) * i / (len(alpha_grad) - 1)
                    # Create transfer mapping scalar value to gradient opacity
                    gotf.AddPoint(xalpha, al)
            elif len(alpha_grad.shape) == 2:  # user passing [(x0,alpha0), ...]
                gotf.AddPoint(vmin, alpha_grad[0][1])
                for xalpha, al in alpha_grad:
                    # Create transfer mapping scalar value to opacity
                    gotf.AddPoint(xalpha, al)
                gotf.AddPoint(vmax, alpha_grad[-1][1])
            # print("alpha_grad at", round(xalpha, 1), "\tset to", al)
        else:
            gotf.AddPoint(vmin, alpha_grad)  # constant alpha_grad
            gotf.AddPoint(vmax, alpha_grad)
        return self

    def cmap(self, c, alpha=None, vmin=None, vmax=None):
        """Same as `color()`.

        Arguments:
            alpha : (list)
                use a list to specify transparencies along the scalar range
            vmin : (float)
                force the min of the scalar range to be this value
            vmax : (float)
                force the max of the scalar range to be this value
        """
        return self.color(c, alpha, vmin, vmax)

    def jittering(self, status=None):
        """
        If `True`, each ray traversal direction will be perturbed slightly
        using a noise-texture to get rid of wood-grain effects.
        """
        if hasattr(self.mapper, "SetUseJittering"):  # tetmesh doesnt have it
            if status is None:
                return self.mapper.GetUseJittering()
            self.mapper.SetUseJittering(status)
        return self

    def hide_voxels(self, ids):
        """
        Hide voxels (cells) from visualization.

        Example:
            ```python
            from vedo import *
            embryo = Volume(dataurl+'embryo.tif')
            embryo.hide_voxels(list(range(10000)))
            show(embryo, axes=1).close()
            ```

        See also:
            `volume.mask()`
        """
        ghost_mask = np.zeros(self.ncells, dtype=np.uint8)
        ghost_mask[ids] = vtk.vtkDataSetAttributes.HIDDENCELL
        name = vtk.vtkDataSetAttributes.GhostArrayName()
        garr = utils.numpy2vtk(ghost_mask, name=name, dtype=np.uint8)
        self.dataset.GetCellData().AddArray(garr)
        self.dataset.GetCellData().Modified()
        return self


    def mode(self, mode=None):
        """
        Define the volumetric rendering mode following this:
            - 0, composite rendering
            - 1, maximum projection rendering
            - 2, minimum projection rendering
            - 3, average projection rendering
            - 4, additive mode

        The default mode is "composite" where the scalar values are sampled through
        the volume and composited in a front-to-back scheme through alpha blending.
        The final color and opacity is determined using the color and opacity transfer
        functions specified in alpha keyword.

        Maximum and minimum intensity blend modes use the maximum and minimum
        scalar values, respectively, along the sampling ray.
        The final color and opacity is determined by passing the resultant value
        through the color and opacity transfer functions.

        Additive blend mode accumulates scalar values by passing each value
        through the opacity transfer function and then adding up the product
        of the value and its opacity. In other words, the scalar values are scaled
        using the opacity transfer function and summed to derive the final color.
        Note that the resulting image is always grayscale i.e. aggregated values
        are not passed through the color transfer function.
        This is because the final value is a derived value and not a real data value
        along the sampling ray.

        Average intensity blend mode works similar to the additive blend mode where
        the scalar values are multiplied by opacity calculated from the opacity
        transfer function and then added.
        The additional step here is to divide the sum by the number of samples
        taken through the volume.
        As is the case with the additive intensity projection, the final image will
        always be grayscale i.e. the aggregated values are not passed through the
        color transfer function.
        """
        if mode is None:
            return self.mapper.GetBlendMode()

        if isinstance(mode, str):
            if "comp" in mode:
                mode = 0
            elif "proj" in mode:
                if "max" in mode:
                    mode = 1
                elif "min" in mode:
                    mode = 2
                elif "ave" in mode:
                    mode = 3
                else:
                    vedo.logger.warning(f"unknown mode {mode}")
                    mode = 0
            elif "add" in mode:
                mode = 4
            else:
                vedo.logger.warning(f"unknown mode {mode}")
                mode = 0

        self.mapper.SetBlendMode(mode)
        return self

    def shade(self, status=None):
        """
        Set/Get the shading of a Volume.
        Shading can be further controlled with `volume.lighting()` method.

        If shading is turned on, the mapper may perform shading calculations.
        In some cases shading does not apply
        (for example, in maximum intensity projection mode).
        """
        if status is None:
            return self.properties.GetShade()
        self.properties.SetShade(status)
        return self


    def mask(self, data):
        """
        Mask a volume visualization with a binary value.
        Needs to specify keyword mapper='gpu'.

        Example:
        ```python
            from vedo import np, Volume, show
            data_matrix = np.zeros([75, 75, 75], dtype=np.uint8)
            # all voxels have value zero except:
            data_matrix[0:35,   0:35,  0:35] = 1
            data_matrix[35:55, 35:55, 35:55] = 2
            data_matrix[55:74, 55:74, 55:74] = 3
            vol = Volume(data_matrix, c=['white','b','g','r'], mapper='gpu')
            data_mask = np.zeros_like(data_matrix)
            data_mask[10:65, 10:45, 20:75] = 1
            vol.mask(data_mask)
            show(vol, axes=1).close()
        ```
        See also:
            `volume.hide_voxels()`
        """
        mask = Volume(data.astype(np.uint8))
        try:
            self.mapper.SetMaskTypeToBinary()
            self.mapper.SetMaskInput(mask.dataset)
        except AttributeError:
            vedo.logger.error("volume.mask() must create the volume with Volume(..., mapper='gpu')")
        return self

    def interpolation(self, itype):
        """
        Set interpolation type.

        0=nearest neighbour, 1=linear
        """
        self.properties.SetInterpolationType(itype)
        return self


########################################################################################
class PictureVisual(CommonVisual):

    def __init__(self) -> None:
        # print("init PictureVisual")
        super().__init__()

    def memory_size(self):
        """
        Return the size in bytes of the object in memory.
        """
        return self.dataset.GetActualMemorySize()

    def scalar_range(self):
        """
        Return the scalar range of the image.
        """
        return self.dataset.GetScalarRange()

    def alpha(self, a=None):
        """Set/get picture's transparency in the rendering scene."""
        if a is not None:
            self.properties.SetOpacity(a)
            return self
        return self.properties.GetOpacity()

    def level(self, value=None):
        """Get/Set the image color level (brightness) in the rendering scene."""
        if value is None:
            return self.properties.GetColorLevel()
        self.properties.SetColorLevel(value)
        return self

    def window(self, value=None):
        """Get/Set the image color window (contrast) in the rendering scene."""
        if value is None:
            return self.properties.GetColorWindow()
        self.properties.SetColorWindow(value)
        return self

    def bounds(self):
        """Get the bounding box."""
        return self.actor.GetBounds()

    def xbounds(self, i=None):
        """Get the bounds `[xmin,xmax]`. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i is not None:
            return b[i]
        return (b[0], b[1])

    def ybounds(self, i=None):
        """Get the bounds `[ymin,ymax]`. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i == 0:
            return b[2]
        if i == 1:
            return b[3]
        return (b[2], b[3])

    def zbounds(self, i=None):
        """Get the bounds `[zmin,zmax]`. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i == 0:
            return b[4]
        if i == 1:
            return b[5]
        return (b[4], b[5])

    def diagonal_size(self):
        """Get the length of the diagonal of mesh bounding box."""
        b = self.bounds()
        return np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)

    def pos(self, *p):
        """Set/get position of object."""
        if len(p)==0:
            return np.array(self.actor.GetPosition())
        if len(p)==2:
            p = (p[0], p[1], 0)
        self.actor.SetPosition(*p)
        return self

    def origin(self, point=None):
        """Set/get origin of object."""
        if point is None:
            return np.array(self.actor.GetOrigin())
        self.actor.SetOrigin(point)
        return self

    def x(self, x=None):
        """Set/get x coordinate of object."""
        if x is None:
            return self.actor.GetPosition()[0]
        p = self.actor.GetPosition()
        self.actor.SetPosition(x, p[1], p[2])
        return self

    def y(self, y=None):
        """Set/get y coordinate of object."""
        if y is None:
            return self.actor.GetPosition()[1]
        p = self.actor.GetPosition()
        self.actor.SetPosition(p[0], y, p[2])
        return self

    def z(self, z=None):
        """Set/get z coordinate of object."""
        if z is None:
            return self.actor.GetPosition()[2]
        p = self.actor.GetPosition()
        self.actor.SetPosition(p[0], p[1], z)
        return self

    def rotate_x(self, angle):
        """Rotate around x axis."""
        self.actor.RotateX(angle)
        return self

    def rotate_y(self, angle):
        """Rotate around y axis."""
        self.actor.RotateY(angle)
        return self

    def rotate_z(self, angle):
        """Rotate around z axis."""
        self.actor.RotateZ(angle)
        return self

    def reorient(self, old_axis, new_axis):
        """Rotate object to a new orientation."""
        axis = utils.versor(old_axis)
        direction = utils.versor(new_axis)
        angle = np.arccos(np.dot(axis, direction)) * 57.3
        self.actor.RotateWXYZ(angle, np.cross(axis, direction))
        return self

    def shift(self, dp):
        """Add vector to current position."""
        p = self.actor.GetPosition()
        if len(dp)==2:
            dp = (dp[0], dp[1], 0)
        self.actor.SetPosition(p[0] + dp[0], p[1] + dp[1], p[2] + dp[2])
        return self

    def scale(self, s=None, absolute=False):
        """Set/get scaling factor."""
        if s is None:
            return np.array(self.actor.GetScale())
        if absolute:
            self.actor.SetScale(s, s, s)
        else:
            self.actor.SetScale(np.array(self.actor.GetScale()) * s)
        return self


########################################################################################
class Actor2D(vtk.vtkActor2D):
    """Wrapping of `vtkActor2D`."""

    def __init__(self):
        """Manage 2D objects."""
        super().__init__()

        self.mapper = None
        self.properties = self.GetProperty()
        self.filename = ""

    def layer(self, value=None):
        """Set/Get the layer number in the overlay planes into which to render."""
        if value is None:
            return self.GetLayerNumber()
        self.SetLayerNumber(value)
        return self

    def pos(self, px=None, py=None):
        """Set/Get the screen-coordinate position."""
        if isinstance(px, str):
            vedo.logger.error("Use string descriptors only inside the constructor")
            return self
        if px is None:
            return np.array(self.GetPosition(), dtype=int)
        if py is not None:
            p = [px, py]
        else:
            p = px
        assert len(p) == 2, "Error: len(pos) must be 2 for Actor2D"
        self.SetPosition(p)
        return self

    def coordinate_system(self, value=None):
        """
        Set/get the coordinate system which this coordinate is defined in.

        The options are:
            0. Display
            1. Normalized Display
            2. Viewport
            3. Normalized Viewport
            4. View
            5. Pose
            6. World
        """
        coor = self.GetPositionCoordinate()
        if value is None:
            return coor.GetCoordinateSystem()
        coor.SetCoordinateSystem(value)
        return self

    def on(self):
        """Set object visibility."""
        self.VisibilityOn()
        return self

    def off(self):
        """Set object visibility."""
        self.VisibilityOn()
        return self

    def toggle(self):
        """Toggle object visibility."""
        self.SetVisibility(not self.GetVisibility())
        return self

    def pickable(self, value=True):
        """Set object pickability."""
        self.SetPickable(value)
        return self

    def alpha(self, value=None):
        """Set/Get the object opacity."""
        if value is None:
            return self.properties.GetOpacity()
        self.properties.SetOpacity(value)
        return self

    def ps(self, point_size=None):
        if point_size is None:
            return self.properties.GetPointSize()
        self.properties.SetPointSize(point_size)
        return self

    def ontop(self, value=True):
        """Keep the object always on top of everything else."""
        if value:
            self.properties.SetDisplayLocationToForeground()
        else:
            self.properties.SetDisplayLocationToBackground()
        return self

    def add_observer(self, event_name, func, priority=0):
        """Add a callback function that will be called when an event occurs."""
        event_name = utils.get_vtk_name_event(event_name)
        idd = self.AddObserver(event_name, func, priority)
        return idd
