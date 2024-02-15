#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import sys
import time
import numpy as np

import vedo.vtkclasses as vtki  # a wrapper for lazy imports

import vedo
from vedo import transformations
from vedo import settings
from vedo import utils
from vedo import backends
from vedo import addons


__docformat__ = "google"

__doc__ = """
This module defines the main class Plotter to manage objects and 3D rendering.

![](https://vedo.embl.es/images/basic/multirenderers.png)
"""

__all__ = ["Plotter", "show", "close"]

########################################################################################
class Event:
    """
    This class holds the info from an event in the window, works as dictionary too.
    """

    __slots__ = [
        "name",
        "title",
        "id",
        "timerid",
        "time",
        "priority",
        "at",
        "object",
        "actor",
        "picked3d",
        "keypress",
        "picked2d",
        "delta2d",
        "angle2d",
        "speed2d",
        "delta3d",
        "speed3d",
        "isPoints",
        "isMesh",
        "isAssembly",
        "isVolume",
        "isImage",
        "isActor2D",
    ]

    def __init__(self):
        self.name = "event"
        self.title = ""
        self.id = 0
        self.timerid = 0
        self.time = 0
        self.priority = 0
        self.at = 0
        self.object = None
        self.actor = None
        self.picked3d = ()
        self.keypress = ""
        self.picked2d = ()
        self.delta2d = ()
        self.angle2d = 0
        self.speed2d = ()
        self.delta3d = ()
        self.speed3d = 0
        self.isPoints = False
        self.isMesh = False
        self.isAssembly = False
        self.isVolume = False
        self.isImage = False
        self.isActor2D = False

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(id(self))})".ljust(75),
            bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m"
        for n in self.__slots__:
            if n == "actor":
                continue
            out += f"{n}".ljust(11) + ": "
            val = str(self[n]).replace("\n", "")[:65].rstrip()
            if val == "True":
                out += "\x1b[32;1m"
            elif val == "False":
                out += "\x1b[31;1m"
            out += val + "\x1b[0m\n"
        return out.rstrip()

    def keys(self):
        return self.__slots__


##############################################################################################
def show(
    *objects,
    at=None,
    shape=(1, 1),
    N=None,
    pos=(0, 0),
    size="auto",
    screensize="auto",
    title="vedo",
    bg="white",
    bg2=None,
    axes=None,
    interactive=None,
    offscreen=False,
    sharecam=True,
    resetcam=True,
    zoom=None,
    viewup="",
    azimuth=0.0,
    elevation=0.0,
    roll=0.0,
    camera=None,
    mode=None,
    screenshot="",
    new=False,
):
    """
    Create on the fly an instance of class Plotter and show the object(s) provided.

    Arguments:
        at : (int)
            number of the renderer to plot to, in case of more than one exists
        shape : (list, str)
            Number of sub-render windows inside of the main window. E.g.:
            specify two across with shape=(2,1) and a two by two grid
            with shape=(2, 2). By default there is only one renderer.

            Can also accept a shape as string descriptor. E.g.:
            - shape="3|1" means 3 plots on the left and 1 on the right,
            - shape="4/2" means 4 plots on top of 2 at bottom.
        N : (int)
            number of desired sub-render windows arranged automatically in a grid
        pos : (list)
            position coordinates of the top-left corner of the rendering window
            on the screen
        size : (list)
            size of the rendering window
        screensize : (list)
            physical size of the monitor screen
        title : (str)
            window title
        bg : (color)
            background color or specify jpg image file name with path
        bg2 : (color)
            background color of a gradient towards the top
        axes : (int)
            set the type of axes to be shown:
            - 0,  no axes
            - 1,  draw three gray grid walls
            - 2,  show cartesian axes from (0,0,0)
            - 3,  show positive range of cartesian axes from (0,0,0)
            - 4,  show a triad at bottom left
            - 5,  show a cube at bottom left
            - 6,  mark the corners of the bounding box
            - 7,  draw a 3D ruler at each side of the cartesian axes
            - 8,  show the `vtkCubeAxesActor` object
            - 9,  show the bounding box outLine
            - 10, show three circles representing the maximum bounding box
            - 11, show a large grid on the x-y plane
            - 12, show polar axes
            - 13, draw a simple ruler at the bottom of the window
            - 14: draw a `CameraOrientationWidget`

            Axis type-1 can be fully customized by passing a dictionary.
            Check `vedo.addons.Axes()` for the full list of options.
        azimuth/elevation/roll : (float)
            move camera accordingly the specified value
        viewup : (str, list)
            either `['x', 'y', 'z']` or a vector to set vertical direction
        resetcam : (bool)
            re-adjust camera position to fit objects
        camera : (dict, vtkCamera)
            camera parameters can further be specified with a dictionary
            assigned to the `camera` keyword (E.g. `show(camera={'pos':(1,2,3), 'thickness':1000,})`):
            - **pos** (list),  the position of the camera in world coordinates
            - **focal_point** (list), the focal point of the camera in world coordinates
            - **viewup** (list), the view up direction for the camera
            - **distance** (float), set the focal point to the specified distance from the camera position.
            - **clipping_range** (float), distance of the near and far clipping planes along the direction of projection.
            - **parallel_scale** (float),
            scaling used for a parallel projection, i.e. the height of the viewport
            in world-coordinate distances. The default is 1. Note that the "scale" parameter works as
            an "inverse scale", larger numbers produce smaller images.
            This method has no effect in perspective projection mode.
            - **thickness** (float),
            set the distance between clipping planes. This method adjusts the far clipping
            plane to be set a distance 'thickness' beyond the near clipping plane.
            - **view_angle** (float),
            the camera view angle, which is the angular height of the camera view
            measured in degrees. The default angle is 30 degrees.
            This method has no effect in parallel projection mode.
            The formula for setting the angle up for perfect perspective viewing is:
            angle = 2*atan((h/2)/d) where h is the height of the RenderWindow
            (measured by holding a ruler up to your screen) and d is the distance
            from your eyes to the screen.
        interactive : (bool)
            pause and interact with window (True) or continue execution (False)
        rate : (float)
            maximum rate of `show()` in Hertz
        mode : (int, str)
            set the type of interaction:
            - 0 = TrackballCamera [default]
            - 1 = TrackballActor
            - 2 = JoystickCamera
            - 3 = JoystickActor
            - 4 = Flight
            - 5 = RubberBand2D
            - 6 = RubberBand3D
            - 7 = RubberBandZoom
            - 8 = Terrain
            - 9 = Unicam
            - 10 = Image
        new : (bool)
            if set to `True`, a call to show will instantiate
            a new Plotter object (a new window) instead of reusing the first created.
            If new is `True`, but the existing plotter was instantiated with a different
            argument for `offscreen`, `new` is ignored and a new Plotter is created anyway.
    """
    if len(objects) == 0:
        objects = None
    elif len(objects) == 1:
        objects = objects[0]
    else:
        objects = utils.flatten(objects)

    # If a plotter instance is already present, check if the offscreen argument
    # is the same as the one requested by the user. If not, create a new
    # plotter instance (see https://github.com/marcomusy/vedo/issues/1026)
    if vedo.plotter_instance and vedo.plotter_instance.offscreen != offscreen:
        new = True

    if vedo.plotter_instance and not new:  # Plotter exists
        plt = vedo.plotter_instance

    else:  # Plotter must be created

        if utils.is_sequence(at):  # user passed a sequence for "at"

            if not utils.is_sequence(objects):
                vedo.logger.error("in show() input must be a list.")
                raise RuntimeError()
            if len(at) != len(objects):
                vedo.logger.error("in show() lists 'input' and 'at' must have equal lengths")
                raise RuntimeError()
            if shape == (1, 1) and N is None:
                N = max(at) + 1

        elif at is None and (N or shape != (1, 1)):

            if not utils.is_sequence(objects):
                e = "in show(), N or shape is set, but input is not a sequence\n"
                e += "              you may need to specify e.g. at=0"
                vedo.logger.error(e)
                raise RuntimeError()
            at = list(range(len(objects)))

        plt = Plotter(
            shape=shape,
            N=N,
            pos=pos,
            size=size,
            screensize=screensize,
            title=title,
            axes=axes,
            sharecam=sharecam,
            resetcam=resetcam,
            interactive=interactive,
            offscreen=offscreen,
            bg=bg,
            bg2=bg2,
        )

    if settings.dry_run_mode >= 2:
        return plt

    # use _plt_to_return because plt.show() can return a k3d plot
    _plt_to_return = None

    if utils.is_sequence(at):

        for i, act in enumerate(objects):
            _plt_to_return = plt.show(
                act,
                at=i,
                zoom=zoom,
                resetcam=resetcam,
                viewup=viewup,
                azimuth=azimuth,
                elevation=elevation,
                roll=roll,
                camera=camera,
                interactive=False,
                mode=mode,
                screenshot=screenshot,
                bg=bg,
                bg2=bg2,
                axes=axes,
            )

        if (
            interactive
            or len(at) == N
            or (isinstance(shape[0], int) and len(at) == shape[0] * shape[1])
        ):
            # note that shape can be a string
            if plt.interactor and not offscreen and (interactive is None or interactive):
                plt.interactor.Start()
                if plt._must_close_now:
                    plt.interactor.GetRenderWindow().Finalize()
                    plt.interactor.TerminateApp()
                    plt.interactor = None
                    plt.window = None
                    plt.renderer = None
                    plt.renderers = []
                    plt.camera = None
                    
    else:

        _plt_to_return = plt.show(
            objects,
            at=at,
            zoom=zoom,
            resetcam=resetcam,
            viewup=viewup,
            azimuth=azimuth,
            elevation=elevation,
            roll=roll,
            camera=camera,
            interactive=interactive,
            mode=mode,
            screenshot=screenshot,
            bg=bg,
            bg2=bg2,
            axes=axes,
        )

    return _plt_to_return


def close():
    """Close the last created Plotter instance if it exists."""
    if not vedo.plotter_instance:
        return
    vedo.plotter_instance.close()
    return


########################################################################
class Plotter:
    """Main class to manage objects."""

    def __init__(
        self,
        shape=(1, 1),
        N=None,
        pos=(0, 0),
        size="auto",
        screensize="auto",
        title="vedo",
        bg="white",
        bg2=None,
        axes=None,
        sharecam=True,
        resetcam=True,
        interactive=None,
        offscreen=False,
        qt_widget=None,
        wx_widget=None,
    ):
        """
        Arguments:
            shape : (str, list)
                shape of the grid of renderers in format (rows, columns). Ignored if N is specified.
            N : (int)
                number of desired renderers arranged in a grid automatically.
            pos : (list)
                (x,y) position in pixels of top-left corner of the rendering window on the screen
            size : (str, list)
                size of the rendering window. If 'auto', guess it based on screensize.
            screensize : (list)
                physical size of the monitor screen in pixels
            bg : (color, str)
                background color or specify jpg image file name with path
            bg2 : (color)
                background color of a gradient towards the top
            title : (str)
                window title

            axes : (int)

                Note that Axes type-1 can be fully customized by passing a dictionary `axes=dict()`.
                Check out `vedo.addons.Axes()` for the available options.

                    - 0,  no axes
                    - 1,  draw three gray grid walls
                    - 2,  show cartesian axes from (0,0,0)
                    - 3,  show positive range of cartesian axes from (0,0,0)
                    - 4,  show a triad at bottom left
                    - 5,  show a cube at bottom left
                    - 6,  mark the corners of the bounding box
                    - 7,  draw a 3D ruler at each side of the cartesian axes
                    - 8,  show the VTK CubeAxesActor object
                    - 9,  show the bounding box outLine
                    - 10, show three circles representing the maximum bounding box
                    - 11, show a large grid on the x-y plane (use with zoom=8)
                    - 12, show polar axes
                    - 13, draw a simple ruler at the bottom of the window
                    - 14: draw a camera orientation widget

            sharecam : (bool)
                if False each renderer will have an independent camera
            interactive : (bool)
                if True will stop after show() to allow interaction with the 3d scene
            offscreen : (bool)
                if True will not show the rendering window
            qt_widget : (QVTKRenderWindowInteractor)
                render in a Qt-Widget using an QVTKRenderWindowInteractor.
                See examples `qt_windows[1,2,3].py` and `qt_cutter.py`.
        """
        vedo.plotter_instance = self

        if interactive is None:
            interactive = bool(N in (0, 1, None) and shape == (1, 1))
        self._interactive = interactive
        # print("interactive", interactive, N, shape)

        self.objects = []           # list of objects to be shown
        self.clicked_object = None  # holds the object that has been clicked
        self.clicked_actor = None   # holds the actor that has been clicked

        self.shape = shape   # nr. of subwindows in grid
        self.axes = axes     # show axes type nr.
        self.title = title   # window title
        self.size = size     # window size
        self.backgrcol = bg  # used also by backend notebooks

        self.offscreen= offscreen
        self.resetcam = resetcam
        self.sharecam = sharecam  # share the same camera if multiple renderers
        self.pos      = pos       # used by vedo.file_io

        self.picker   = None  # hold the vtkPicker object
        self.picked2d = None  # 2d coords of a clicked point on the rendering window
        self.picked3d = None  # 3d coords of a clicked point on an actor

        self.qt_widget = qt_widget  # QVTKRenderWindowInteractor
        self.wx_widget = wx_widget  # wxVTKRenderWindowInteractor
        self.interactor = None
        self.window = None
        self.renderer = None
        self.renderers = []  # list of renderers

        # mostly internal stuff:
        self.hover_legends = []
        self.justremoved = None
        self.axes_instances = []
        self.clock = 0
        self.sliders = []
        self.buttons = []
        self.widgets = []
        self.cutter_widget = None
        self.hint_widget = None
        self.background_renderer = None
        self.last_event = None
        self.skybox = None
        self._icol = 0
        self._clockt0 = time.time()
        self._extralight = None
        self._cocoa_initialized = False
        self._cocoa_process_events = True  # make one call in show()
        self._must_close_now = False

        #####################################################################
        if settings.default_backend == "2d":
            self.offscreen = True
            if self.size == "auto":
                self.size = (800, 600)

        elif settings.default_backend == "k3d":
            if self.size == "auto":
                self.size = (1000, 1000)
            ####################################
            return  ############################
            ####################################

        #############################################################
        if settings.default_backend in ["vtk", "2d", "trame"]:

            if screensize == "auto":
                screensize = (2160, 1440)  # TODO: get actual screen size

            # build the rendering window:
            self.window = vtki.vtkRenderWindow()

            self.window.GlobalWarningDisplayOff()

            if self.title == "vedo":  # check if dev version
                if "dev" in vedo.__version__:
                    self.title = f"vedo ({vedo.__version__})"
            self.window.SetWindowName(self.title)

            # more settings
            if settings.use_depth_peeling:
                self.window.SetAlphaBitPlanes(settings.alpha_bit_planes)
            self.window.SetMultiSamples(settings.multi_samples)

            self.window.SetPolygonSmoothing(settings.polygon_smoothing)
            self.window.SetLineSmoothing(settings.line_smoothing)
            self.window.SetPointSmoothing(settings.point_smoothing)

        #############################################################
        if N:  # N = number of renderers. Find out the best

            if shape != (1, 1):  # arrangement based on minimum nr. of empty renderers
                vedo.logger.warning("having set N, shape is ignored.")

            x, y = screensize
            nx = int(np.sqrt(int(N * y / x) + 1))
            ny = int(np.sqrt(int(N * x / y) + 1))
            lm = [
                (nx, ny),
                (nx, ny + 1),
                (nx - 1, ny),
                (nx + 1, ny),
                (nx, ny - 1),
                (nx - 1, ny + 1),
                (nx + 1, ny - 1),
                (nx + 1, ny + 1),
                (nx - 1, ny - 1),
            ]
            ind, minl = 0, 1000
            for i, m in enumerate(lm):
                l = m[0] * m[1]
                if N <= l < minl:
                    ind = i
                    minl = l
            shape = lm[ind]

        ##################################################
        if isinstance(shape, str):

            if "|" in shape:
                if self.size == "auto":
                    self.size = (800, 1200)
                n = int(shape.split("|")[0])
                m = int(shape.split("|")[1])
                rangen = reversed(range(n))
                rangem = reversed(range(m))
            else:
                if self.size == "auto":
                    self.size = (1200, 800)
                m = int(shape.split("/")[0])
                n = int(shape.split("/")[1])
                rangen = range(n)
                rangem = range(m)

            if n >= m:
                xsplit = m / (n + m)
            else:
                xsplit = 1 - n / (n + m)
            if settings.window_splitting_position:
                xsplit = settings.window_splitting_position

            for i in rangen:
                arenderer = vtki.vtkRenderer()
                if "|" in shape:
                    arenderer.SetViewport(0, i / n, xsplit, (i + 1) / n)
                else:
                    arenderer.SetViewport(i / n, 0, (i + 1) / n, xsplit)
                self.renderers.append(arenderer)

            for i in rangem:
                arenderer = vtki.vtkRenderer()

                if "|" in shape:
                    arenderer.SetViewport(xsplit, i / m, 1, (i + 1) / m)
                else:
                    arenderer.SetViewport(i / m, xsplit, (i + 1) / m, 1)
                self.renderers.append(arenderer)

            for r in self.renderers:
                r.SetLightFollowCamera(settings.light_follows_camera)

                r.SetUseDepthPeeling(settings.use_depth_peeling)
                # r.SetUseDepthPeelingForVolumes(settings.use_depth_peeling)
                if settings.use_depth_peeling:
                    r.SetMaximumNumberOfPeels(settings.max_number_of_peels)
                    r.SetOcclusionRatio(settings.occlusion_ratio)
                r.SetUseFXAA(settings.use_fxaa)
                r.SetPreserveDepthBuffer(settings.preserve_depth_buffer)

                r.SetBackground(vedo.get_color(self.backgrcol))

                self.axes_instances.append(None)

            self.shape = (n + m,)

        elif utils.is_sequence(shape) and isinstance(shape[0], dict):
            # passing a sequence of dicts for renderers specifications

            if self.size == "auto":
                self.size = (1000, 800)

            for rd in shape:
                x0, y0 = rd["bottomleft"]
                x1, y1 = rd["topright"]
                bg_ = rd.pop("bg", "white")
                bg2_ = rd.pop("bg2", None)

                arenderer = vtki.vtkRenderer()
                arenderer.SetLightFollowCamera(settings.light_follows_camera)

                arenderer.SetUseDepthPeeling(settings.use_depth_peeling)
                # arenderer.SetUseDepthPeelingForVolumes(settings.use_depth_peeling)
                if settings.use_depth_peeling:
                    arenderer.SetMaximumNumberOfPeels(settings.max_number_of_peels)
                    arenderer.SetOcclusionRatio(settings.occlusion_ratio)
                arenderer.SetUseFXAA(settings.use_fxaa)
                arenderer.SetPreserveDepthBuffer(settings.preserve_depth_buffer)

                arenderer.SetViewport(x0, y0, x1, y1)
                arenderer.SetBackground(vedo.get_color(bg_))
                if bg2_:
                    arenderer.GradientBackgroundOn()
                    arenderer.SetBackground2(vedo.get_color(bg2_))

                self.renderers.append(arenderer)
                self.axes_instances.append(None)

            self.shape = (len(shape),)

        else:

            if isinstance(self.size, str) and self.size == "auto":
                # figure out a reasonable window size
                f = 1.5
                x, y = screensize
                xs = y / f * shape[1]  # because y<x
                ys = y / f * shape[0]
                if xs > x / f:  # shrink
                    xs = x / f
                    ys = xs / shape[1] * shape[0]
                if ys > y / f:
                    ys = y / f
                    xs = ys / shape[0] * shape[1]
                self.size = (int(xs), int(ys))
                if shape == (1, 1):
                    self.size = (int(y / f), int(y / f))  # because y<x
            else:
                self.size = (self.size[0], self.size[1])

            try:
                image_actor = None
                bgname = str(self.backgrcol).lower()
                if ".jpg" in bgname or ".jpeg" in bgname or ".png" in bgname:
                    self.window.SetNumberOfLayers(2)
                    self.background_renderer = vtki.vtkRenderer()
                    self.background_renderer.SetLayer(0)
                    self.background_renderer.InteractiveOff()
                    self.background_renderer.SetBackground(vedo.get_color(bg2))
                    image_actor = vedo.Image(self.backgrcol).actor
                    self.window.AddRenderer(self.background_renderer)
                    self.background_renderer.AddActor(image_actor)
            except AttributeError:
                pass

            for i in reversed(range(shape[0])):
                for j in range(shape[1]):
                    arenderer = vtki.vtkRenderer()
                    arenderer.SetLightFollowCamera(settings.light_follows_camera)
                    arenderer.SetTwoSidedLighting(settings.two_sided_lighting)

                    arenderer.SetUseDepthPeeling(settings.use_depth_peeling)
                    # arenderer.SetUseDepthPeelingForVolumes(settings.use_depth_peeling)
                    if settings.use_depth_peeling:
                        arenderer.SetMaximumNumberOfPeels(settings.max_number_of_peels)
                        arenderer.SetOcclusionRatio(settings.occlusion_ratio)
                    arenderer.SetUseFXAA(settings.use_fxaa)
                    arenderer.SetPreserveDepthBuffer(settings.preserve_depth_buffer)

                    if image_actor:
                        arenderer.SetLayer(1)

                    arenderer.SetBackground(vedo.get_color(self.backgrcol))
                    if bg2:
                        arenderer.GradientBackgroundOn()
                        arenderer.SetBackground2(vedo.get_color(bg2))

                    x0 = i / shape[0]
                    y0 = j / shape[1]
                    x1 = (i + 1) / shape[0]
                    y1 = (j + 1) / shape[1]
                    arenderer.SetViewport(y0, x0, y1, x1)
                    self.renderers.append(arenderer)
                    self.axes_instances.append(None)
            self.shape = shape

        if self.renderers:
            self.renderer = self.renderers[0]
            self.camera.SetParallelProjection(settings.use_parallel_projection)

        #########################################################
        if self.qt_widget or self.wx_widget:
            if self.qt_widget:
                self.window = self.qt_widget.GetRenderWindow()  # overwrite
            else:
                self.window = self.wx_widget.GetRenderWindow()
            self.interactor = self.window.GetInteractor()

        #########################################################
        for r in self.renderers:
            self.window.AddRenderer(r)
            # set the background gradient if any
            if settings.background_gradient_orientation > 0:
                try:
                    modes = [
                        vtki.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL,
                        vtki.vtkViewport.GradientModes.VTK_GRADIENT_HORIZONTAL,
                        vtki.vtkViewport.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_SIDE,
                        vtki.vtkViewport.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_CORNER,
                    ]
                    r.SetGradientMode(modes[settings.background_gradient_orientation])
                    r.GradientBackgroundOn()
                except AttributeError:
                    pass

        #########################################################
        if self.qt_widget or self.wx_widget:
            # self.window.SetSize(int(self.size[0]), int(self.size[1]))
            self.interactor.SetRenderWindow(self.window)
            # vsty = vtki.new("InteractorStyleTrackballCamera")
            # self.interactor.SetInteractorStyle(vsty)
            if settings.enable_default_keyboard_callbacks:
                self.interactor.AddObserver("KeyPressEvent", self._default_keypress)
            if settings.enable_default_mouse_callbacks:
                self.interactor.AddObserver("LeftButtonPressEvent", self._default_mouseleftclick)
            return  ################
            ########################

        if self.size[0] == "f":  # full screen
            self.size = "fullscreen"
            self.window.SetFullScreen(True)
            self.window.BordersOn()
        else:
            self.window.SetSize(int(self.size[0]), int(self.size[1]))

        if self.offscreen:
            if self.axes in (4, 5, 8, 12, 14):
                self.axes = 0  # does not work with those
            self.window.SetOffScreenRendering(True)
            self.interactor = None
            self._interactive = False
            return  ################
            ########################

        self.window.SetPosition(pos)

        #########################################################
        self.interactor = vtki.vtkRenderWindowInteractor()

        self.interactor.SetRenderWindow(self.window)
        vsty = vtki.new("InteractorStyleTrackballCamera")
        self.interactor.SetInteractorStyle(vsty)
        self.interactor.RemoveObservers("CharEvent")

        if settings.enable_default_keyboard_callbacks:
            self.interactor.AddObserver("KeyPressEvent", self._default_keypress)
        if settings.enable_default_mouse_callbacks:
            self.interactor.AddObserver("LeftButtonPressEvent", self._default_mouseleftclick)

    ##################################################################### ..init ends here.

    def __str__(self):
        """Return Plotter info."""
        axtype = {
            0: "(no axes)",
            1: "(default customizable grid walls)",
            2: "(cartesian axes from origin",
            3: "(positive range of cartesian axes from origin",
            4: "(axes triad at bottom left)",
            5: "(oriented cube at bottom left)",
            6: "(mark the corners of the bounding box)",
            7: "(3D ruler at each side of the cartesian axes)",
            8: "(the vtkCubeAxesActor object)",
            9: "(the bounding box outline)",
            10: "(circles of maximum bounding box range)",
            11: "(show a large grid on the x-y plane)",
            12: "(show polar axes)",
            13: "(simple ruler at the bottom of the window)",
            14: "(the vtkCameraOrientationWidget object)",
        }

        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(id(self))})".ljust(75),
            bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m"
        if self.interactor:
            out += "window title".ljust(14) + ": " + self.title + "\n"
            out += "window size".ljust(14) + f": {self.window.GetSize()}"
            out += f", full_screen={self.window.GetScreenSize()}\n"
            out += "activ renderer".ljust(14) + ": nr." + str(self.renderers.index(self.renderer))
            out += f" (out of {len(self.renderers)} renderers)\n"

        bns, totpt = [], 0
        for a in self.objects:
            try:
                b = a.bounds()
                bns.append(b)
            except AttributeError:
                pass
            try:
                totpt += a.npoints
            except AttributeError:
                pass
        out += "n. of objects".ljust(14) + f": {len(self.objects)}"
        out += f" ({totpt} vertices)\n" if totpt else "\n"

        if len(bns) > 0:
            min_bns = np.min(bns, axis=0)
            max_bns = np.max(bns, axis=0)
            bx1, bx2 = utils.precision(min_bns[0], 3), utils.precision(max_bns[1], 3)
            by1, by2 = utils.precision(min_bns[2], 3), utils.precision(max_bns[3], 3)
            bz1, bz2 = utils.precision(min_bns[4], 3), utils.precision(max_bns[5], 3)
            out += "bounds".ljust(14) + ":"
            out += " x=(" + bx1 + ", " + bx2 + "),"
            out += " y=(" + by1 + ", " + by2 + "),"
            out += " z=(" + bz1 + ", " + bz2 + ")\n"

        if utils.is_integer(self.axes):
            out += "axes style".ljust(14) + f": {self.axes} {axtype[self.axes]}\n"
        elif isinstance(self.axes, dict):
            out += "axes style".ljust(14) + f": 1 {axtype[1]}\n"
        else:
            out += "axes style".ljust(14) + f": {[self.axes]}\n"
        return out.rstrip() + "\x1b[0m"

    def print(self):
        """Print information about the current instance."""
        print(self.__str__())
        return self

    def __iadd__(self, objects):
        self.add(objects)
        return self

    def __isub__(self, objects):
        self.remove(objects)
        return self

    def __enter__(self):
        # context manager like in "with Plotter() as plt:"
        return self

    def __exit__(self, *args, **kwargs):
        # context manager like in "with Plotter() as plt:"
        self.close()

    def initialize_interactor(self):
        """Initialize the interactor if not already initialized."""
        if self.offscreen:
            return self
        if self.interactor:
            if not self.interactor.GetInitialized():
                self.interactor.Initialize()
                self.interactor.RemoveObservers("CharEvent")
        return self

    def process_events(self):
        """Process all pending events."""
        self.initialize_interactor()
        if self.interactor:
            try:
                self.interactor.ProcessEvents()
            except AttributeError:
                pass
        return self

    def at(self, nren, yren=None):
        """
        Select the current renderer number as an int.
        Can also use the `[nx, ny]` format.
        """
        if yren is not None:
            nren = (yren) * self.shape[1] + (nren)
            if nren < 0 or nren > len(self.renderers):
                vedo.logger.error(f"at({nren, yren}) is malformed!")
                raise RuntimeError

        self.renderer = self.renderers[nren]
        return self

    def add(self, *objs, at=None):
        """
        Append the input objects to the internal list of objects to be shown.

        Arguments:
            at : (int)
                add the object at the specified renderer
        """
        if at is not None:
            ren = self.renderers[at]
        else:
            ren = self.renderer

        objs = utils.flatten(objs)
        for ob in objs:
            if ob and ob not in self.objects:
                self.objects.append(ob)

        acts = self._scan_input_return_acts(objs)

        for a in acts:

            if ren:

                if isinstance(a, vtki.vtkInteractorObserver):
                    a.add_to(self)  # from cutters
                    continue

                try:
                    ren.AddActor(a)
                except TypeError:
                    ren.AddActor(a.actor)

                if hasattr(a, "rendered_at"):
                    ir = self.renderers.index(ren)
                    a.rendered_at.add(ir)
                if isinstance(a, vtki.vtkFollower):
                    a.SetCamera(self.camera)
                if isinstance(a, vedo.visual.LightKit):
                    a.lightkit.AddLightsToRenderer(ren)

        return self

    def remove(self, *objs, at=None):
        """
        Remove input object to the internal list of objects to be shown.

        Objects to be removed can be referenced by their assigned name,

        Arguments:
            at : (int)
                remove the object at the specified renderer
        """
        # TODO and you can also use wildcards like `*` and `?`.
        if at is not None:
            ren = self.renderers[at]
        else:
            ren = self.renderer

        objs = [ob for ob in utils.flatten(objs) if ob]

        has_str = False
        for ob in objs:
            if isinstance(ob, str):
                has_str = True
                break

        has_actor = False
        for ob in objs:
            if hasattr(ob, "actor") and ob.actor:
                has_actor = True
                break

        if has_str or has_actor:
            # need to get the actors to search for
            for a in self.get_actors(include_non_pickables=True):
                # print("PARSING", [a])
                try:
                    if (a.name and a.name in objs) or a in objs:
                        objs.append(a)
                    # if a.name:
                    #     bools = [utils.parse_pattern(ob, a.name)[0] for ob in objs]
                    #     if any(bools) or a in objs:
                    #         objs.append(a)
                    #     print('a.name',a.name, objs,any(bools))
                except AttributeError:  # no .name
                    # passing the actor so get back the object with .retrieve_object()
                    try:
                        vobj = a.retrieve_object()
                        if (vobj.name and vobj.name in objs) or vobj in objs:
                            # print('vobj.name', vobj.name)
                            objs.append(vobj)
                    except AttributeError:
                        pass

        ir = self.renderers.index(ren)

        ids = []
        for ob in set(objs):

            # will remove it from internal list if possible
            try:
                idx = self.objects.index(ob)
                ids.append(idx)
            except ValueError:
                pass

            if ren:  ### remove it from the renderer

                if isinstance(ob, vtki.vtkInteractorObserver):
                    ob.remove_from(self)  # from cutters
                    continue

                try:
                    ren.RemoveActor(ob)
                except TypeError:
                    try:
                        ren.RemoveActor(ob.actor)
                    except AttributeError:
                        pass

                if hasattr(ob, "rendered_at"):
                    ob.rendered_at.discard(ir)

                if hasattr(ob, "scalarbar") and ob.scalarbar:
                    ren.RemoveActor(ob.scalarbar)
                if hasattr(ob, "_caption") and ob._caption:
                    ren.RemoveActor(ob._caption)
                if hasattr(ob, "shadows") and ob.shadows:
                    for sha in ob.shadows:
                        ren.RemoveActor(sha.actor)
                if hasattr(ob, "trail") and ob.trail:
                    ren.RemoveActor(ob.trail.actor)
                    ob.trail_points = []
                    if hasattr(ob.trail, "shadows") and ob.trail.shadows:
                        for sha in ob.trail.shadows:
                            ren.RemoveActor(sha.actor)

                elif isinstance(ob, vedo.visual.LightKit):
                    ob.lightkit.RemoveLightsFromRenderer(ren)

        # for i in ids: # WRONG way of doing it!
        #     del self.objects[i]
        # instead we do:
        self.objects = [ele for i, ele in enumerate(self.objects) if i not in ids]
        return self

    @property
    def actors(self):
        """Return the list of actors."""
        return [ob.actor for ob in self.objects if hasattr(ob, "actor")]

    def remove_lights(self):
        """Remove all the present lights in the current renderer."""
        if self.renderer:
            self.renderer.RemoveAllLights()
        return self

    def pop(self, at=None):
        """
        Remove the last added object from the rendering window.
        This method is typically used in loops or callback functions.
        """
        if at is not None and not isinstance(at, int):
            # wrong usage pitfall
            vedo.logger.error("argument of pop() must be an integer")
            raise RuntimeError()

        if self.objects:
            self.remove(self.objects[-1], at)
        return self

    def render(self, resetcam=False):
        """Render the scene. This method is typically used in loops or callback functions."""

        if settings.dry_run_mode >= 2:
            return self

        if not self.window:
            return self

        self.initialize_interactor()

        if resetcam:
            self.renderer.ResetCamera()

        self.window.Render()

        if self._cocoa_process_events and self.interactor.GetInitialized():
            if "Darwin" in vedo.sys_platform and not self.offscreen:
                self.interactor.ProcessEvents()
                self._cocoa_process_events = False
        return self

    def interactive(self):
        """
        Start window interaction.
        Analogous to `show(..., interactive=True)`.
        """
        if settings.dry_run_mode >= 1:
            return self
        self.initialize_interactor()
        if self.interactor:
            # print("self.interactor.Start()")
            self.interactor.Start()
            # print("self.interactor.Start() done")
            if self._must_close_now:
                # print("self.interactor.TerminateApp()")
                self.interactor.GetRenderWindow().Finalize()
                self.interactor.TerminateApp()
                self.interactor = None
                self.window = None
                self.renderer = None
                self.renderers = []
                self.camera = None
        return self

    def use_depth_peeling(self, at=None, value=True):
        """
        Specify whether use depth peeling algorithm at this specific renderer
        Call this method before the first rendering.
        """
        if at is None:
            ren = self.renderer
        else:
            ren = self.renderers[at]
        ren.SetUseDepthPeeling(value)
        return self

    def background(self, c1=None, c2=None, at=None, mode=0):
        """Set the color of the background for the current renderer.
        A different renderer index can be specified by keyword `at`.

        Arguments:
            c1 : (list)
                background main color.
            c2 : (list)
                background color for the upper part of the window.
            at : (int)
                renderer index.
            mode : (int)
                background mode (needs vtk version >= 9.3)
                    0 = vertical,
                    1 = horizontal,
                    2 = radial farthest side,
                    3 = radia farthest corner.
        """
        if not self.renderers:
            return self
        if at is None:
            r = self.renderer
        else:
            r = self.renderers[at]

        if c1 is None and c2 is None:
            return np.array(r.GetBackground())

        if r:
            if c1 is not None:
                r.SetBackground(vedo.get_color(c1))
            if c2 is not None:
                r.GradientBackgroundOn()
                r.SetBackground2(vedo.get_color(c2))
                if mode:
                    try:  # only works with vtk>=9.3
                        modes = [
                            vtki.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL,
                            vtki.vtkViewport.GradientModes.VTK_GRADIENT_HORIZONTAL,
                            vtki.vtkViewport.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_SIDE,
                            vtki.vtkViewport.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_CORNER,
                        ]
                        r.SetGradientMode(modes[settings.background_gradient_orientation])
                    except AttributeError:
                        pass

            else:
                r.GradientBackgroundOff()
        return self

    ##################################################################
    def get_meshes(self, at=None, include_non_pickables=False, unpack_assemblies=True):
        """
        Return a list of Meshes from the specified renderer.

        Arguments:
            at : (int)
                specify which renderer to look at.
            include_non_pickables : (bool)
                include non-pickable objects
            unpack_assemblies : (bool)
                unpack assemblies into their components
        """
        if at is None:
            renderer = self.renderer
            at = self.renderers.index(renderer)
        elif isinstance(at, int):
            renderer = self.renderers[at]

        has_global_axes = False
        if isinstance(self.axes_instances[at], vedo.Assembly):
            has_global_axes = True

        if unpack_assemblies:
            acs = renderer.GetActors()
        else:
            acs = renderer.GetViewProps()

        objs = []
        acs.InitTraversal()
        for _ in range(acs.GetNumberOfItems()):

            if unpack_assemblies:
                a = acs.GetNextItem()
            else:
                a = acs.GetNextProp()

            if isinstance(a, vtki.vtkVolume):
                continue

            if include_non_pickables or a.GetPickable():
                if a == self.axes_instances[at]:
                    continue
                if has_global_axes and a in self.axes_instances[at].actors:
                    continue
                try:
                    objs.append(a.retrieve_object())
                except AttributeError:
                    pass
        return objs

    def get_volumes(self, at=None, include_non_pickables=False):
        """
        Return a list of Volumes from the specified renderer.

        Arguments:
            at : (int)
                specify which renderer to look at
            include_non_pickables : (bool)
                include non-pickable objects
        """
        if at is None:
            renderer = self.renderer
            at = self.renderers.index(renderer)
        elif isinstance(at, int):
            renderer = self.renderers[at]

        vols = []
        acs = renderer.GetVolumes()
        acs.InitTraversal()
        for _ in range(acs.GetNumberOfItems()):
            a = acs.GetNextItem()
            if include_non_pickables or a.GetPickable():
                try:
                    vols.append(a.retrieve_object())
                except AttributeError:
                    pass
        return vols

    def get_actors(self, at=None, include_non_pickables=False):
        """
        Return a list of Volumes from the specified renderer.

        Arguments:
            at : (int)
                specify which renderer to look at
            include_non_pickables : (bool)
                include non-pickable objects
        """
        if at is None:
            renderer = self.renderer
            at = self.renderers.index(renderer)
        elif isinstance(at, int):
            renderer = self.renderers[at]

        acts = []
        acs = renderer.GetViewProps()
        acs.InitTraversal()
        for _ in range(acs.GetNumberOfItems()):
            a = acs.GetNextProp()
            if include_non_pickables or a.GetPickable():
                acts.append(a)
        return acts
    
    def check_actors_trasform(self, at=None):
        """
        Reset the transformation matrix of all actors at specified renderer.
        This is only useful when actors have been moved/rotated/scaled manually
        in an already rendered scene using interactors like
        'TrackballActor' or 'JoystickActor'.
        """
        # see issue https://github.com/marcomusy/vedo/issues/1046
        for a in self.get_actors(at=at, include_non_pickables=True):
            try:
                M = a.GetMatrix()
            except AttributeError:
                continue
            if M and not M.IsIdentity():
                try:
                    a.retrieve_object().apply_transform_from_actor()
                    # vedo.logger.info(
                    #     f"object '{a.retrieve_object().name}' "
                    #     "was manually moved. Updated to its current position."
                    # )
                except AttributeError:
                    pass
        return self

    def reset_camera(self, tight=None):
        """
        Reset the camera position and zooming.
        If tight (float) is specified the zooming reserves a padding space
        in the xy-plane expressed in percent of the average size.
        """
        if tight is None:
            self.renderer.ResetCamera()
        else:
            x0, x1, y0, y1, z0, z1 = self.renderer.ComputeVisiblePropBounds()

            cam = self.renderer.GetActiveCamera()

            self.renderer.ComputeAspect()
            aspect = self.renderer.GetAspect()
            angle = np.pi * cam.GetViewAngle() / 180.0
            dx, dy = (x1 - x0) * 0.999, (y1 - y0) * 0.999
            dist = max(dx / aspect[0], dy) / np.sin(angle / 2) / 2

            cam.SetViewUp(0, 1, 0)
            cam.SetPosition(x0 + dx / 2, y0 + dy / 2, dist * (1 + tight))
            cam.SetFocalPoint(x0 + dx / 2, y0 + dy / 2, 0)
            if cam.GetParallelProjection():
                ps = max(dx / aspect[0], dy) / 2
                cam.SetParallelScale(ps * (1 + tight))
            self.renderer.ResetCameraClippingRange(x0, x1, y0, y1, z0, z1)
        return self

    def reset_viewup(self, smooth=True):
        """
        Reset the orientation of the camera to the closest orthogonal direction and view-up.
        """
        vbb = addons.compute_visible_bounds()[0]
        x0, x1, y0, y1, z0, z1 = vbb
        mx, my, mz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        d = self.camera.GetDistance()

        viewups = np.array(
            [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)]
        )
        positions = np.array(
            [
                (mx, my, mz + d),
                (mx, my, mz - d),
                (mx, my + d, mz),
                (mx, my - d, mz),
                (mx + d, my, mz),
                (mx - d, my, mz),
            ]
        )

        vu = np.array(self.camera.GetViewUp())
        vui = np.argmin(np.linalg.norm(viewups - vu, axis=1))

        poc = np.array(self.camera.GetPosition())
        foc = np.array(self.camera.GetFocalPoint())
        a = poc - foc
        b = positions - foc
        a = a / np.linalg.norm(a)
        b = b.T * (1 / np.linalg.norm(b, axis=1))
        pui = np.argmin(np.linalg.norm(b.T - a, axis=1))

        if smooth:
            outtimes = np.linspace(0, 1, num=11, endpoint=True)
            for t in outtimes:
                vv = vu * (1 - t) + viewups[vui] * t
                pp = poc * (1 - t) + positions[pui] * t
                ff = foc * (1 - t) + np.array([mx, my, mz]) * t
                self.camera.SetViewUp(vv)
                self.camera.SetPosition(pp)
                self.camera.SetFocalPoint(ff)
                self.render()

            # interpolator does not respect parallel view...:
            # cam1 = dict(
            #     pos=poc,
            #     viewup=vu,
            #     focal_point=(mx,my,mz),
            #     clipping_range=self.camera.GetClippingRange()
            # )
            # # cam1 = self.camera
            # cam2 = dict(
            #     pos=positions[pui],
            #     viewup=viewups[vui],
            #     focal_point=(mx,my,mz),
            #     clipping_range=self.camera.GetClippingRange()
            # )
            # vcams = self.move_camera([cam1, cam2], output_times=outtimes, smooth=0)
            # for c in vcams:
            #     self.renderer.SetActiveCamera(c)
            #     self.render()
        else:

            self.camera.SetViewUp(viewups[vui])
            self.camera.SetPosition(positions[pui])
            self.camera.SetFocalPoint(mx, my, mz)

        self.renderer.ResetCameraClippingRange()

        # vbb, _, _, _ = addons.compute_visible_bounds()
        # x0,x1, y0,y1, z0,z1 = vbb
        # self.renderer.ResetCameraClippingRange(x0, x1, y0, y1, z0, z1)
        self.render()
        return self

    def move_camera(self, cameras, t=0, times=(), smooth=True, output_times=()):
        """
        Takes as input two cameras set camera at an interpolated position:

        Cameras can be vtkCamera or dictionaries in format:

            `dict(pos=..., focal_point=..., viewup=..., distance=..., clipping_range=...)`

        Press `shift-C` key in interactive mode to dump a python snipplet
        of parameters for the current camera view.
        """
        nc = len(cameras)
        if len(times) == 0:
            times = np.linspace(0, 1, num=nc, endpoint=True)

        assert len(times) == nc

        cin = vtki.new("CameraInterpolator")

        # cin.SetInterpolationTypeToLinear() # buggy?
        if nc > 2 and smooth:
            cin.SetInterpolationTypeToSpline()

        for i, cam in enumerate(cameras):
            vcam = cam
            if isinstance(cam, dict):
                vcam = utils.camera_from_dict(cam)
            cin.AddCamera(times[i], vcam)

        mint, maxt = cin.GetMinimumT(), cin.GetMaximumT()
        rng = maxt - mint

        if len(output_times) == 0:
            cin.InterpolateCamera(t * rng, self.camera)
            self.renderer.SetActiveCamera(self.camera)
            return [self.camera]
        else:
            vcams = []
            for tt in output_times:
                c = vtki.vtkCamera()
                cin.InterpolateCamera(tt * rng, c)
                vcams.append(c)
            return vcams

    def fly_to(self, point):
        """
        Fly camera to the specified point.

        Arguments:
            point : (list)
                point in space to place camera.

        Example:
            ```python
            from vedo import *
            cone = Cone()
            plt = Plotter(axes=1)
            plt.show(cone)
            plt.fly_to([1,0,0])
            plt.interactive().close()
            ```
        """
        if self.interactor:
            self.resetcam = False
            self.interactor.FlyTo(self.renderer, point)
        return self

    def look_at(self, plane="xy"):
        """Move the camera so that it looks at the specified cartesian plane"""
        cam = self.renderer.GetActiveCamera()
        fp = np.array(cam.GetFocalPoint())
        p = np.array(cam.GetPosition())
        dist = np.linalg.norm(fp - p)
        plane = plane.lower()
        if "x" in plane and "y" in plane:
            cam.SetPosition(fp[0], fp[1], fp[2] + dist)
            cam.SetViewUp(0.0, 1.0, 0.0)
        elif "x" in plane and "z" in plane:
            cam.SetPosition(fp[0], fp[1] - dist, fp[2])
            cam.SetViewUp(0.0, 0.0, 1.0)
        elif "y" in plane and "z" in plane:
            cam.SetPosition(fp[0] + dist, fp[1], fp[2])
            cam.SetViewUp(0.0, 0.0, 1.0)
        else:
            vedo.logger.error(f"in plotter.look() cannot understand argument {plane}")
        return self

    def record(self, filename=""):
        """
        Record camera, mouse, keystrokes and all other events.
        Recording can be toggled on/off by pressing key "R".

        Arguments:
            filename : (str)
                ascii file to store events.
                The default is `settings.cache_directory+"vedo/recorded_events.log"`.

        Returns:
            a string descriptor of events.

        Examples:
            - [record_play.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/record_play.py)
        """
        if settings.dry_run_mode >= 1:
            return self
        if not self.interactor:
            vedo.logger.warning("Cannot record events, no interactor defined.")
            return self
        erec = vtki.new("InteractorEventRecorder")
        erec.SetInteractor(self.interactor)
        if not filename:
            if not os.path.exists(settings.cache_directory):
                os.makedirs(settings.cache_directory)
            home_dir = os.path.expanduser("~")
            filename = os.path.join(
                home_dir, settings.cache_directory, "vedo", "recorded_events.log")
            print("Events will be recorded in", filename)
        erec.SetFileName(filename)
        erec.SetKeyPressActivationValue("R")
        erec.EnabledOn()
        erec.Record()
        self.interactor.Start()
        erec.Stop()
        erec.EnabledOff()
        with open(filename, "r", encoding="UTF-8") as fl:
            events = fl.read()
        erec = None
        return events

    def play(self, recorded_events="", repeats=0):
        """
        Play camera, mouse, keystrokes and all other events.

        Arguments:
            events : (str)
                file o string of events.
                The default is `settings.cache_directory+"vedo/recorded_events.log"`.
            repeats : (int)
                number of extra repeats of the same events. The default is 0.

        Examples:
            - [record_play.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/record_play.py)
        """
        if settings.dry_run_mode >= 1:
            return self
        if not self.interactor:
            vedo.logger.warning("Cannot play events, no interactor defined.")
            return self

        erec = vtki.new("InteractorEventRecorder")
        erec.SetInteractor(self.interactor)

        if not recorded_events:
            home_dir = os.path.expanduser("~")
            recorded_events = os.path.join(
                home_dir, settings.cache_directory, "vedo", "recorded_events.log")

        if recorded_events.endswith(".log"):
            erec.ReadFromInputStringOff()
            erec.SetFileName(recorded_events)
        else:
            erec.ReadFromInputStringOn()
            erec.SetInputString(recorded_events)

        erec.Play()
        for _ in range(repeats):
            erec.Rewind()
            erec.Play()
        erec.EnabledOff()
        erec = None
        return self

    def parallel_projection(self, value=True, at=None):
        """
        Use parallel projection `at` a specified renderer.
        Object is seen from "infinite" distance, e.i. remove any perspective effects.
        An input value equal to -1 will toggle it on/off.
        """
        if at is not None:
            r = self.renderers[at]
        else:
            r = self.renderer
        if value == -1:
            val = r.GetActiveCamera().GetParallelProjection()
            value = not val
        r.GetActiveCamera().SetParallelProjection(value)
        r.Modified()
        return self

    def render_hidden_lines(self, value=True):
        """Remove hidden lines when in wireframe mode."""
        self.renderer.SetUseHiddenLineRemoval(not value)
        return self

    def fov(self, angle):
        """
        Set the field of view angle for the camera.
        This is the angle of the camera frustum in the horizontal direction.
        High values will result in a wide-angle lens (fish-eye effect),
        and low values will result in a telephoto lens.

        Default value is 30 degrees.
        """
        self.renderer.GetActiveCamera().UseHorizontalViewAngleOn()
        self.renderer.GetActiveCamera().SetViewAngle(angle)
        return self

    def zoom(self, zoom):
        """Apply a zooming factor for the current camera view"""
        self.renderer.GetActiveCamera().Zoom(zoom)
        return self

    def azimuth(self, angle):
        """Rotate camera around the view up vector."""
        self.renderer.GetActiveCamera().Azimuth(angle)
        return self

    def elevation(self, angle):
        """Rotate the camera around the cross product of the negative
        of the direction of projection and the view up vector."""
        self.renderer.GetActiveCamera().Elevation(angle)
        return self

    def roll(self, angle):
        """Roll the camera about the direction of projection."""
        self.renderer.GetActiveCamera().Roll(angle)
        return self

    def dolly(self, value):
        """Move the camera towards (value>0) or away from (value<0) the focal point."""
        self.renderer.GetActiveCamera().Dolly(value)
        return self

    ##################################################################
    def add_slider(
        self,
        sliderfunc,
        xmin,
        xmax,
        value=None,
        pos=4,
        title="",
        font="Calco",
        title_size=1,
        c=None,
        alpha=1,
        show_value=True,
        delayed=False,
        **options,
    ):
        """
        Add a `vedo.addons.Slider2D` which can call an external custom function.

        Arguments:
            sliderfunc : (Callable)
                external function to be called by the widget
            xmin : (float)
                lower value of the slider
            xmax : (float)
                upper value
            value : (float)
                current value
            pos : (list, str)
                position corner number: horizontal [1-5] or vertical [11-15]
                it can also be specified by corners coordinates [(x1,y1), (x2,y2)]
                and also by a string descriptor (eg. "bottom-left")
            title : (str)
                title text
            font : (str)
                title font face. Check [available fonts here](https://vedo.embl.es/fonts).
            title_size : (float)
                title text scale [1.0]
            show_value : (bool)
                if True current value is shown
            delayed : (bool)
                if True the callback is delayed until when the mouse button is released
            alpha : (float)
                opacity of the scalar bar texts
            slider_length : (float)
                slider length
            slider_width : (float)
                slider width
            end_cap_length : (float)
                length of the end cap
            end_cap_width : (float)
                width of the end cap
            tube_width : (float)
                width of the tube
            title_height : (float)
                width of the title
            tformat : (str)
                format of the title

        Examples:
            - [sliders1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders1.py)
            - [sliders2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders2.py)

            ![](https://user-images.githubusercontent.com/32848391/50738848-be033480-11d8-11e9-9b1a-c13105423a79.jpg)
        """
        if c is None:  # automatic black or white
            c = (0.8, 0.8, 0.8)
            if np.sum(vedo.get_color(self.backgrcol)) > 1.5:
                c = (0.2, 0.2, 0.2)
        else:
            c = vedo.get_color(c)

        slider2d = addons.Slider2D(
            sliderfunc,
            xmin,
            xmax,
            value,
            pos,
            title,
            font,
            title_size,
            c,
            alpha,
            show_value,
            delayed,
            **options,
        )

        if self.renderer:
            slider2d.renderer = self.renderer
            if self.interactor:
                slider2d.interactor = self.interactor
                slider2d.on()
                self.sliders.append([slider2d, sliderfunc])
        return slider2d

    def add_slider3d(
        self,
        sliderfunc,
        pos1,
        pos2,
        xmin,
        xmax,
        value=None,
        s=0.03,
        t=1,
        title="",
        rotation=0.0,
        c=None,
        show_value=True,
    ):
        """
        Add a 3D slider widget which can call an external custom function.

        Arguments:
            sliderfunc : (function)
                external function to be called by the widget
            pos1 : (list)
                first position 3D coordinates
            pos2 : (list)
                second position coordinates
            xmin : (float)
                lower value
            xmax : (float)
                upper value
            value : (float)
                initial value
            s : (float)
                label scaling factor
            t : (float)
                tube scaling factor
            title : (str)
                title text
            c : (color)
                slider color
            rotation : (float)
                title rotation around slider axis
            show_value : (bool)
                if True current value is shown

        Examples:
            - [sliders3d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders3d.py)

            ![](https://user-images.githubusercontent.com/32848391/52859555-4efcf200-312d-11e9-9290-6988c8295163.png)
        """
        if c is None:  # automatic black or white
            c = (0.8, 0.8, 0.8)
            if np.sum(vedo.get_color(self.backgrcol)) > 1.5:
                c = (0.2, 0.2, 0.2)
        else:
            c = vedo.get_color(c)

        slider3d = addons.Slider3D(
            sliderfunc,
            pos1,
            pos2,
            xmin,
            xmax,
            value,
            s,
            t,
            title,
            rotation,
            c,
            show_value,
        )
        slider3d.renderer = self.renderer
        slider3d.interactor = self.interactor
        slider3d.on()
        self.sliders.append([slider3d, sliderfunc])
        return slider3d

    def add_button(
        self,
        fnc=None,
        states=("On", "Off"),
        c=("w", "w"),
        bc=("green4", "red4"),
        pos=(0.7, 0.1),
        size=24,
        font="Courier",
        bold=True,
        italic=False,
        alpha=1,
        angle=0,
    ):
        """
        Add a button to the renderer window.

        Arguments:
            states : (list)
                a list of possible states, e.g. ['On', 'Off']
            c : (list)
                a list of colors for each state
            bc : (list)
                a list of background colors for each state
            pos : (list)
                2D position from left-bottom corner
            size : (float)
                size of button font
            font : (str)
                font type. Check [available fonts here](https://vedo.embl.es/fonts).
            bold : (bool)
                bold font face (False)
            italic : (bool)
                italic font face (False)
            alpha : (float)
                opacity level
            angle : (float)
                anticlockwise rotation in degrees

        Returns:
            `vedo.addons.Button` object.

        Examples:
            - [buttons1.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/buttons1.py)
            - [buttons2.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/buttons2.py)

            ![](https://user-images.githubusercontent.com/32848391/50738870-c0fe2500-11d8-11e9-9b78-92754f5c5968.jpg)
        """
        if self.interactor:
            bu = addons.Button(fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle)
            self.renderer.AddActor2D(bu)
            self.buttons.append(bu)
            # bu.function_id = self.add_callback("LeftButtonPress", bu.function) # not good
            bu.function_id = bu.add_observer("pick", bu.function, priority=10)
            return bu

    def add_spline_tool(
        self, points, pc="k", ps=8, lc="r4", ac="g5",
        lw=2, alpha=1, closed=False, ontop=True, can_add_nodes=True,
    ):
        """
        Add a spline tool to the current plotter.
        Nodes of the spline can be dragged in space with the mouse.
        Clicking on the line itself adds an extra point.
        Selecting a point and pressing del removes it.

        Arguments:
            points : (Mesh, Points, array)
                the set of vertices forming the spline nodes.
            pc : (str)
                point color. The default is 'k'.
            ps : (str)
                point size. The default is 8.
            lc : (str)
                line color. The default is 'r4'.
            ac : (str)
                active point marker color. The default is 'g5'.
            lw : (int)
                line width. The default is 2.
            alpha : (float)
                line transparency.
            closed : (bool)
                spline is meant to be closed. The default is False.

        Returns:
            a `SplineTool` object.

        Examples:
            - [spline_tool.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/spline_tool.py)

            ![](https://vedo.embl.es/images/basic/spline_tool.png)
        """
        sw = addons.SplineTool(points, pc, ps, lc, ac, lw, alpha, closed, ontop, can_add_nodes)
        sw.interactor = self.interactor
        sw.on()
        sw.Initialize(sw.points.dataset)
        sw.representation.SetRenderer(self.renderer)
        sw.representation.SetClosedLoop(closed)
        sw.representation.BuildRepresentation()
        self.widgets.append(sw)
        return sw

    def add_icon(self, icon, pos=3, size=0.08):
        """Add an inset icon mesh into the same renderer.

        Arguments:
            pos : (int, list)
                icon position in the range [1-4] indicating one of the 4 corners,
                or it can be a tuple (x,y) as a fraction of the renderer size.
            size : (float)
                size of the square inset.

        Examples:
            - [icon.py](https://github.com/marcomusy/vedo/tree/master/examples/other/icon.py)
        """
        iconw = addons.Icon(icon, pos, size)

        iconw.SetInteractor(self.interactor)
        iconw.EnabledOn()
        iconw.InteractiveOff()
        self.widgets.append(iconw)
        return iconw

    def add_global_axes(self, axtype=None, c=None):
        """Draw axes on scene. Available axes types:

        Arguments:
            axtype : (int)
                - 0,  no axes,
                - 1,  draw three gray grid walls
                - 2,  show cartesian axes from (0,0,0)
                - 3,  show positive range of cartesian axes from (0,0,0)
                - 4,  show a triad at bottom left
                - 5,  show a cube at bottom left
                - 6,  mark the corners of the bounding box
                - 7,  draw a 3D ruler at each side of the cartesian axes
                - 8,  show the vtkCubeAxesActor object
                - 9,  show the bounding box outLine
                - 10, show three circles representing the maximum bounding box
                - 11, show a large grid on the x-y plane
                - 12, show polar axes
                - 13, draw a simple ruler at the bottom of the window

            Axis type-1 can be fully customized by passing a dictionary axes=dict().

        Example:
            ```python
            from vedo import Box, show
            b = Box(pos=(0, 0, 0), length=80, width=90, height=70).alpha(0.1)
            show(
                b,
                axes={
                    "xtitle": "Some long variable [a.u.]",
                    "number_of_divisions": 4,
                    # ...
                },
            )
            ```

        Examples:
            - [custom_axes1.py](https://github.com/marcomusy/vedo/blob/master/examples/pyplot/custom_axes1.py)
            - [custom_axes2.py](https://github.com/marcomusy/vedo/blob/master/examples/pyplot/custom_axes2.py)
            - [custom_axes3.py](https://github.com/marcomusy/vedo/blob/master/examples/pyplot/custom_axes3.py)
            - [custom_axes4.py](https://github.com/marcomusy/vedo/blob/master/examples/pyplot/custom_axes4.py)

            <img src="https://user-images.githubusercontent.com/32848391/72752870-ab7d5280-3bc3-11ea-8911-9ace00211e23.png" width="600">
        """
        addons.add_global_axes(axtype, c)
        return self

    def add_legend_box(self, **kwargs):
        """Add a legend to the top right.

        Examples:
            - [legendbox.py](https://github.com/marcomusy/vedo/blob/master/examples/examples/basic/legendbox.py),
            - [flag_labels1.py](https://github.com/marcomusy/vedo/blob/master/examples/examples/other/flag_labels1.py)
            - [flag_labels2.py](https://github.com/marcomusy/vedo/blob/master/examples/examples/other/flag_labels2.py)
        """
        acts = self.get_meshes()
        lb = addons.LegendBox(acts, **kwargs)
        self.add(lb)
        return self

    def add_hint(
        self,
        obj,
        text="",
        c="k",
        bg="yellow9",
        font="Calco",
        size=18,
        justify=0,
        angle=0,
        delay=250,
    ):
        """
        Create a pop-up hint style message when hovering an object.
        Use `add_hint(obj, False)` to disable a hinting a specific object.
        Use `add_hint(None)` to disable all hints.

        Arguments:
            obj : (Mesh, Points)
                the object to associate the pop-up to
            text : (str)
                string description of the pop-up
            delay : (int)
                milliseconds to wait before pop-up occurs
        """
        if self.offscreen or not self.interactor:
            return self

        if vedo.vtk_version[:2] == (9, 0) and "Linux" in vedo.sys_platform:
            # Linux vtk9.0 is bugged
            vedo.logger.warning(
                f"add_hint() is not available on Linux platforms for vtk{vedo.vtk_version}."
            )
            return self

        if obj is None:
            self.hint_widget.EnabledOff()
            self.hint_widget.SetInteractor(None)
            self.hint_widget = None
            return self

        if text is False and self.hint_widget:
            self.hint_widget.RemoveBalloon(obj)
            return self

        if text == "":
            if obj.name:
                text = obj.name
            elif obj.filename:
                text = obj.filename
            else:
                return self

        if not self.hint_widget:
            self.hint_widget = vtki.vtkBalloonWidget()

            rep = self.hint_widget.GetRepresentation()
            rep.SetBalloonLayoutToImageRight()

            trep = rep.GetTextProperty()
            trep.SetFontFamily(vtki.VTK_FONT_FILE)
            trep.SetFontFile(utils.get_font_path(font))
            trep.SetFontSize(size)
            trep.SetColor(vedo.get_color(c))
            trep.SetBackgroundColor(vedo.get_color(bg))
            trep.SetShadow(0)
            trep.SetJustification(justify)
            trep.UseTightBoundingBoxOn()

            self.hint_widget.ManagesCursorOff()
            self.hint_widget.SetTimerDuration(delay)
            self.hint_widget.SetInteractor(self.interactor)
            if angle:
                trep.SetOrientation(angle)
                trep.SetBackgroundOpacity(0)
            # else:
            #     trep.SetBackgroundOpacity(0.5) # doesnt work well
            self.hint_widget.SetRepresentation(rep)
            self.widgets.append(self.hint_widget)
            self.hint_widget.EnabledOn()

        bst = self.hint_widget.GetBalloonString(obj.actor)
        if bst:
            self.hint_widget.UpdateBalloonString(obj.actor, text)
        else:
            self.hint_widget.AddBalloon(obj.actor, text)

        return self

    def add_shadows(self):
        """Add shadows at the current renderer."""
        if self.renderer:
            shadows = vtki.new("ShadowMapPass")
            seq = vtki.new("SequencePass")
            passes = vtki.new("RenderPassCollection")
            passes.AddItem(shadows.GetShadowMapBakerPass())
            passes.AddItem(shadows)
            seq.SetPasses(passes)
            camerapass = vtki.new("CameraPass")
            camerapass.SetDelegatePass(seq)
            self.renderer.SetPass(camerapass)
        return self

    def add_ambient_occlusion(self, radius, bias=0.01, blur=True, samples=100):
        """
        Screen Space Ambient Occlusion.

        For every pixel on the screen, the pixel shader samples the depth values around
        the current pixel and tries to compute the amount of occlusion from each of the sampled
        points.

        Arguments:
            radius : (float)
                radius of influence in absolute units
            bias : (float)
                bias of the normals
            blur : (bool)
                add a blurring to the sampled positions
            samples : (int)
                number of samples to probe

        Examples:
            - [ssao.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/ssao.py)

            ![](https://vedo.embl.es/images/basic/ssao.jpg)
        """
        lights = vtki.new("LightsPass")

        opaque = vtki.new("OpaquePass")

        ssaoCam = vtki.new("CameraPass")
        ssaoCam.SetDelegatePass(opaque)

        ssao = vtki.new("SSAOPass")
        ssao.SetRadius(radius)
        ssao.SetBias(bias)
        ssao.SetBlur(blur)
        ssao.SetKernelSize(samples)
        ssao.SetDelegatePass(ssaoCam)

        translucent = vtki.new("TranslucentPass")

        volpass = vtki.new("VolumetricPass")
        ddp = vtki.new("DualDepthPeelingPass")
        ddp.SetTranslucentPass(translucent)
        ddp.SetVolumetricPass(volpass)

        over = vtki.new("OverlayPass")

        collection = vtki.new("RenderPassCollection")
        collection.AddItem(lights)
        collection.AddItem(ssao)
        collection.AddItem(ddp)
        collection.AddItem(over)

        sequence = vtki.new("SequencePass")
        sequence.SetPasses(collection)

        cam = vtki.new("CameraPass")
        cam.SetDelegatePass(sequence)

        self.renderer.SetPass(cam)
        return self

    def add_depth_of_field(self, autofocus=True):
        """Add a depth of field effect in the scene."""
        lights = vtki.new("LightsPass")

        opaque = vtki.new("OpaquePass")

        dofCam = vtki.new("CameraPass")
        dofCam.SetDelegatePass(opaque)

        dof = vtki.new("DepthOfFieldPass")
        dof.SetAutomaticFocalDistance(autofocus)
        dof.SetDelegatePass(dofCam)

        collection = vtki.new("RenderPassCollection")
        collection.AddItem(lights)
        collection.AddItem(dof)

        sequence = vtki.new("SequencePass")
        sequence.SetPasses(collection)

        cam = vtki.new("CameraPass")
        cam.SetDelegatePass(sequence)

        self.renderer.SetPass(cam)
        return self

    def _add_skybox(self, hdrfile):
        # many hdr files are at https://polyhaven.com/all

        reader = vtki.new("HDRReader")
        # Check the image can be read.
        if not reader.CanReadFile(hdrfile):
            vedo.logger.error(f"Cannot read HDR file {hdrfile}")
            return self
        reader.SetFileName(hdrfile)
        reader.Update()

        texture = vtki.vtkTexture()
        texture.SetColorModeToDirectScalars()
        texture.SetInputData(reader.GetOutput())

        # Convert to a cube map
        tcm = vtki.new("EquirectangularToCubeMapTexture")
        tcm.SetInputTexture(texture)
        # Enable mipmapping to handle HDR image
        tcm.MipmapOn()
        tcm.InterpolateOn()

        self.renderer.SetEnvironmentTexture(tcm)
        self.renderer.UseImageBasedLightingOn()
        self.skybox = vtki.new("Skybox")
        self.skybox.SetTexture(tcm)
        self.renderer.AddActor(self.skybox)
        return self

    def add_renderer_frame(self, c=None, alpha=None, lw=None, padding=None):
        """
        Add a frame to the renderer subwindow.

        Arguments:
            c : (color)
                color name or index
            alpha : (float)
                opacity level
            lw : (int)
                line width in pixels.
            padding : (float)
                padding space in pixels.
        """
        if c is None:  # automatic black or white
            c = (0.9, 0.9, 0.9)
            if np.sum(vedo.plotter_instance.renderer.GetBackground()) > 1.5:
                c = (0.1, 0.1, 0.1)
        renf = addons.RendererFrame(c, alpha, lw, padding)
        if renf:
            self.renderer.AddActor(renf)
        return self

    def add_hover_legend(
        self,
        at=None,
        c=None,
        pos="bottom-left",
        font="Calco",
        s=0.75,
        bg="auto",
        alpha=0.1,
        maxlength=24,
        use_info=False,
    ):
        """
        Add a legend with 2D text which is triggered by hovering the mouse on an object.

        The created text object are stored in `plotter.hover_legends`.

        Returns:
            the id of the callback function.

        Arguments:
            c : (color)
                Text color. If None then black or white is chosen automatically
            pos : (str)
                text positioning
            font : (str)
                text font type. Check [available fonts here](https://vedo.embl.es/fonts).
            s : (float)
                text size scale
            bg : (color)
                background color of the 2D box containing the text
            alpha : (float)
                box transparency
            maxlength : (int)
                maximum number of characters per line
            use_info : (bool)
                visualize the content of the `obj.info` attribute

        Examples:
            - [hover_legend.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/hover_legend.py)
            - [earthquake_browser.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/earthquake_browser.py)

            ![](https://vedo.embl.es/images/pyplot/earthquake_browser.jpg)
        """
        hoverlegend = vedo.shapes.Text2D(pos=pos, font=font, c=c, s=s, alpha=alpha, bg=bg)

        if at is None:
            at = self.renderers.index(self.renderer)

        def _legfunc(evt):
            if not evt.object or not self.renderer or at != evt.at:
                if hoverlegend.mapper.GetInput():  # clear and return
                    hoverlegend.mapper.SetInput("")
                    self.render()
                return

            if use_info:
                if hasattr(evt.object, "info"):
                    t = str(evt.object.info)
                else:
                    return
            else:
                t, tp = "", ""
                if evt.isMesh:
                    tp = "Mesh "
                elif evt.isPoints:
                    tp = "Points "
                elif evt.isVolume:
                    tp = "Volume "
                elif evt.isImage:
                    tp = "Image "
                elif evt.isAssembly:
                    tp = "Assembly "
                else:
                    return

                if evt.isAssembly:
                    if not evt.object.name:
                        t += f"Assembly object of {len(evt.object.unpack())} parts\n"
                    else:
                        t += f"Assembly name: {evt.object.name} ({len(evt.object.unpack())} parts)\n"
                else:
                    if evt.object.name:
                        t += f"{tp}name"
                        if evt.isPoints:
                            t += "  "
                        if evt.isMesh:
                            t += "  "
                        t += f": {evt.object.name[:maxlength]}".ljust(maxlength) + "\n"

                if evt.object.filename:
                    t += f"{tp}filename: "
                    t += f"{os.path.basename(evt.object.filename[-maxlength:])}".ljust(maxlength)
                    t += "\n"
                    if not evt.object.file_size:
                        evt.object.file_size, evt.object.created = vedo.file_io.file_info(evt.object.filename)
                    if evt.object.file_size:
                        t += "             : "
                        sz, created = evt.object.file_size, evt.object.created
                        t += f"{created[4:-5]} ({sz})" + "\n"

                if evt.isPoints:
                    indata = evt.object.dataset
                    if indata.GetNumberOfPoints():
                        t += (
                            f"#points/cells: {indata.GetNumberOfPoints()}"
                            f" / {indata.GetNumberOfCells()}"
                        )
                    pdata = indata.GetPointData()
                    cdata = indata.GetCellData()
                    if pdata.GetScalars() and pdata.GetScalars().GetName():
                        t += f"\nPoint array  : {pdata.GetScalars().GetName()}"
                        if pdata.GetScalars().GetName() == evt.object.mapper.GetArrayName():
                            t += " *"
                    if cdata.GetScalars() and cdata.GetScalars().GetName():
                        t += f"\nCell  array  : {cdata.GetScalars().GetName()}"
                        if cdata.GetScalars().GetName() == evt.object.mapper.GetArrayName():
                            t += " *"

                if evt.isImage:
                    t = f"{os.path.basename(evt.object.filename[:maxlength+10])}".ljust(maxlength+10)
                    t += f"\nImage shape: {evt.object.shape}"
                    pcol = self.color_picker(evt.picked2d)
                    t += f"\nPixel color: {vedo.colors.rgb2hex(pcol/255)} {pcol}"

            # change box color if needed in 'auto' mode
            if evt.isPoints and "auto" in str(bg):
                actcol = evt.object.properties.GetColor()
                if hoverlegend.mapper.GetTextProperty().GetBackgroundColor() != actcol:
                    hoverlegend.mapper.GetTextProperty().SetBackgroundColor(actcol)

            # adapt to changes in bg color
            bgcol = self.renderers[at].GetBackground()
            _bgcol = c
            if _bgcol is None:  # automatic black or white
                _bgcol = (0.9, 0.9, 0.9)
                if sum(bgcol) > 1.5:
                    _bgcol = (0.1, 0.1, 0.1)
                if len(set(_bgcol).intersection(bgcol)) < 3:
                    hoverlegend.color(_bgcol)

            if hoverlegend.mapper.GetInput() != t:
                hoverlegend.mapper.SetInput(t)
                self.interactor.Render()
            
            # print("ABORT", idcall, hoverlegend.actor.GetCommand(idcall))
            # hoverlegend.actor.GetCommand(idcall).AbortFlagOn()

        self.add(hoverlegend, at=at)
        self.hover_legends.append(hoverlegend)
        idcall = self.add_callback("MouseMove", _legfunc)
        return idcall

    def add_scale_indicator(
        self,
        pos=(0.7, 0.05),
        s=0.02,
        length=2,
        lw=4,
        c="k1",
        alpha=1,
        units="",
        gap=0.05,
    ):
        """
        Add a Scale Indicator. Only works in parallel mode (no perspective).

        Arguments:
            pos : (list)
                fractional (x,y) position on the screen.
            s : (float)
                size of the text.
            length : (float)
                length of the line.
            units : (str)
                string to show units.
            gap : (float)
                separation of line and text.

        Example:
            ```python
            from vedo import settings, Cube, Plotter
            settings.use_parallel_projection = True # or else it does not make sense!
            cube = Cube().alpha(0.2)
            plt = Plotter(size=(900,600), axes=dict(xtitle='x (um)'))
            plt.add_scale_indicator(units='um', c='blue4')
            plt.show(cube, "Scale indicator with units").close()
            ```
            ![](https://vedo.embl.es/images/feats/scale_indicator.png)
        """
        # Note that this cannot go in addons.py
        # because it needs callbacks and window size
        if not self.interactor:
            return self

        ppoints = vtki.vtkPoints()  # Generate the polyline
        psqr = [[0.0, gap], [length / 10, gap]]
        dd = psqr[1][0] - psqr[0][0]
        for i, pt in enumerate(psqr):
            ppoints.InsertPoint(i, pt[0], pt[1], 0)
        lines = vtki.vtkCellArray()
        lines.InsertNextCell(len(psqr))
        for i in range(len(psqr)):
            lines.InsertCellPoint(i)
        pd = vtki.vtkPolyData()
        pd.SetPoints(ppoints)
        pd.SetLines(lines)

        wsx, wsy = self.window.GetSize()
        if not self.camera.GetParallelProjection():
            vedo.logger.warning("add_scale_indicator called with use_parallel_projection OFF. Skip.")
            return None

        rlabel = vtki.new("VectorText")
        rlabel.SetText("scale")
        tf = vtki.new("TransformPolyDataFilter")
        tf.SetInputConnection(rlabel.GetOutputPort())
        t = vtki.vtkTransform()
        t.Scale(s * wsy / wsx, s, 1)
        tf.SetTransform(t)

        app = vtki.new("AppendPolyData")
        app.AddInputConnection(tf.GetOutputPort())
        app.AddInputData(pd)

        mapper = vtki.new("PolyDataMapper2D")
        mapper.SetInputConnection(app.GetOutputPort())
        cs = vtki.vtkCoordinate()
        cs.SetCoordinateSystem(1)
        mapper.SetTransformCoordinate(cs)

        fractor = vedo.visual.Actor2D()
        csys = fractor.GetPositionCoordinate()
        csys.SetCoordinateSystem(3)
        fractor.SetPosition(pos)
        fractor.SetMapper(mapper)
        fractor.GetProperty().SetColor(vedo.get_color(c))
        fractor.GetProperty().SetOpacity(alpha)
        fractor.GetProperty().SetLineWidth(lw)
        fractor.GetProperty().SetDisplayLocationToForeground()

        def sifunc(iren, ev):
            wsx, wsy = self.window.GetSize()
            ps = self.camera.GetParallelScale()
            newtxt = utils.precision(ps / wsy * wsx * length * dd, 3)
            if units:
                newtxt += " " + units
            if rlabel.GetText() != newtxt:
                rlabel.SetText(newtxt)

        self.renderer.AddActor(fractor)
        self.interactor.AddObserver("MouseWheelBackwardEvent", sifunc)
        self.interactor.AddObserver("MouseWheelForwardEvent", sifunc)
        self.interactor.AddObserver("InteractionEvent", sifunc)
        sifunc(0, 0)
        return fractor

    def fill_event(self, ename="", pos=(), enable_picking=True):
        """
        Create an Event object with information of what was clicked.

        If `enable_picking` is False, no picking will be performed.
        This can be useful to avoid double picking when using buttons.
        """
        if not self.interactor:
            return Event()

        if len(pos) > 0:
            x, y = pos
            self.interactor.SetEventPosition(pos)
        else:
            x, y = self.interactor.GetEventPosition()
        self.renderer = self.interactor.FindPokedRenderer(x, y)

        self.picked2d = (x, y)

        key = self.interactor.GetKeySym()

        if key:
            if "_L" in key or "_R" in key:
                # skip things like Shift_R
                key = ""  # better than None
            else:
                if self.interactor.GetShiftKey():
                    key = key.upper()

                if key == "MINUS":  # fix: vtk9 is ignoring shift chars..
                    key = "underscore"
                elif key == "EQUAL":  # fix: vtk9 is ignoring shift chars..
                    key = "plus"
                elif key == "SLASH":  # fix: vtk9 is ignoring shift chars..
                    key = "?"

                if self.interactor.GetControlKey():
                    key = "Ctrl+" + key

                if self.interactor.GetAltKey():
                    key = "Alt+" + key

        if enable_picking:
            if not self.picker:
                self.picker = vtki.vtkPropPicker()

            self.picker.PickProp(x, y, self.renderer)
            actor = self.picker.GetProp3D()
            # Note that GetProp3D already picks Assembly

            xp, yp = self.interactor.GetLastEventPosition()
            dx, dy = x - xp, y - yp

            delta3d = np.array([0, 0, 0])

            if actor:
                picked3d = np.array(self.picker.GetPickPosition())

                try:
                    vobj = actor.retrieve_object()
                    old_pt = np.asarray(vobj.picked3d)
                    vobj.picked3d = picked3d
                    delta3d = picked3d - old_pt
                except (AttributeError, TypeError):
                    pass

            else:
                picked3d = None

            if not actor:  # try 2D
                actor = self.picker.GetActor2D()

        event = Event()
        event.name = ename
        event.title = self.title
        event.id = -1  # will be set by the timer wrapper function
        event.timerid = -1  # will be set by the timer wrapper function
        event.priority = -1  # will be set by the timer wrapper function
        event.time = time.time()
        event.at = self.renderers.index(self.renderer)
        event.keypress = key
        if enable_picking:
            try:
                event.object = actor.retrieve_object()
            except AttributeError:
                event.object = actor
            try:
                event.actor = actor.retrieve_object()  # obsolete use object instead
            except AttributeError:
                event.actor = actor
            event.picked3d = picked3d
            event.picked2d = (x, y)
            event.delta2d = (dx, dy)
            event.angle2d = np.arctan2(dy, dx)
            event.speed2d = np.sqrt(dx * dx + dy * dy)
            event.delta3d = delta3d
            event.speed3d = np.sqrt(np.dot(delta3d, delta3d))
            event.isPoints = isinstance(event.object, vedo.Points)
            event.isMesh = isinstance(event.object, vedo.Mesh)
            event.isAssembly = isinstance(event.object, vedo.Assembly)
            event.isVolume = isinstance(event.object, vedo.Volume)
            event.isImage = isinstance(event.object, vedo.Image)
            event.isActor2D = isinstance(event.object, vtki.vtkActor2D)
        return event

    def add_callback(self, event_name, func, priority=0.0, enable_picking=True):
        """
        Add a function to be executed while show() is active.

        Return a unique id for the callback.

        The callback function (see example below) exposes a dictionary
        with the following information:
        - `name`: event name,
        - `id`: event unique identifier,
        - `priority`: event priority (float),
        - `interactor`: the interactor object,
        - `at`: renderer nr. where the event occurred
        - `keypress`: key pressed as string
        - `actor`: object picked by the mouse
        - `picked3d`: point picked in world coordinates
        - `picked2d`: screen coords of the mouse pointer
        - `delta2d`: shift wrt previous position (to calculate speed, direction)
        - `delta3d`: ...same but in 3D world coords
        - `angle2d`: angle of mouse movement on screen
        - `speed2d`: speed of mouse movement on screen
        - `speed3d`: speed of picked point in world coordinates
        - `isPoints`: True if of class
        - `isMesh`: True if of class
        - `isAssembly`: True if of class
        - `isVolume`: True if of class Volume
        - `isImage`: True if of class

        If `enable_picking` is False, no picking will be performed.
        This can be useful to avoid double picking when using buttons.

        Frequently used events are:
        - `KeyPress`, `KeyRelease`: listen to keyboard events
        - `LeftButtonPress`, `LeftButtonRelease`: listen to mouse clicks
        - `MiddleButtonPress`, `MiddleButtonRelease`
        - `RightButtonPress`, `RightButtonRelease`
        - `MouseMove`: listen to mouse pointer changing position
        - `MouseWheelForward`, `MouseWheelBackward`
        - `Enter`, `Leave`: listen to mouse entering or leaving the window
        - `Pick`, `StartPick`, `EndPick`: listen to object picking
        - `ResetCamera`, `ResetCameraClippingRange`
        - `Error`, `Warning`
        - `Char`
        - `Timer`

        Check the complete list of events [here](https://vtk.org/doc/nightly/html/classvtkCommand.html).

        Example:
            ```python
            from vedo import *

            def func(evt):
                # this function is called every time the mouse moves
                # (evt is a dotted dictionary)
                if not evt.object:
                    return  # no hit, return
                print("point coords =", evt.picked3d)
                # print(evt) # full event dump

            elli = Ellipsoid()
            plt = Plotter(axes=1)
            plt.add_callback('mouse hovering', func)
            plt.show(elli).close()
            ```

        Examples:
            - [spline_draw.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/spline_draw.py)
            - [colorlines.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/colorlines.py)

                ![](https://vedo.embl.es/images/advanced/spline_draw.png)

            - ..and many others!
        """
        from vtkmodules.util.misc import calldata_type

        if not self.interactor:
            return 0

        if settings.dry_run_mode >= 1:
            return 0

        #########################################
        @calldata_type(vtki.VTK_INT)
        def _func_wrap(iren, ename, timerid=None):
            event = self.fill_event(ename=ename, enable_picking=enable_picking)
            event.timerid = timerid
            event.id = cid
            event.priority = priority
            self.last_event = event
            func(event)

        #########################################

        event_name = utils.get_vtk_name_event(event_name)

        cid = self.interactor.AddObserver(event_name, _func_wrap, priority)
        # print(f"Registering event: {event_name} with id={cid}")
        return cid

    def remove_callback(self, cid):
        """
        Remove a callback function by its id
        or a whole category of callbacks by their name.

        Arguments:
            cid : (int, str)
                Unique id of the callback.
                If an event name is passed all callbacks of that type are removed.
        """
        if self.interactor:
            if isinstance(cid, str):
                cid = utils.get_vtk_name_event(cid)
                self.interactor.RemoveObservers(cid)
            else:
                self.interactor.RemoveObserver(cid)
        return self

    def remove_all_observers(self):
        """
        Remove all observers.

        Example:
        ```python
        from vedo import *

        def kfunc(event):
            print("Key pressed:", event.keypress)
            if event.keypress == 'q':
                plt.close()

        def rfunc(event):
            if event.isImage:
                printc("Right-clicked!", event)
                plt.render()

        img = Image(dataurl+"images/embryo.jpg")

        plt = Plotter(size=(1050, 600))
        plt.parallel_projection(True)
        plt.remove_all_observers()
        plt.add_callback("key press", kfunc)
        plt.add_callback("mouse right click", rfunc)
        plt.show("Right-Click Me! Press q to exit.", img)
        plt.close()
        ```
        """
        if self.interactor:
            self.interactor.RemoveAllObservers()
        return self

    def timer_callback(self, action, timer_id=None, dt=1, one_shot=False):
        """
        Start or stop an existing timer.

        Arguments:
            action : (str)
                Either "create"/"start" or "destroy"/"stop"
            timer_id : (int)
                When stopping the timer, the ID of the timer as returned when created
            dt : (int)
                time in milliseconds between each repeated call
            one_shot : (bool)
                create a one shot timer of prescribed duration instead of a repeating one

        Examples:
            - [timer_callback1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/timer_callback1.py)
            - [timer_callback2.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/timer_callback2.py)

            ![](https://vedo.embl.es/images/advanced/timer_callback1.jpg)
        """
        if action in ("create", "start"):
            if timer_id is not None:
                vedo.logger.warning("you set a timer_id but it will be ignored.")
            if one_shot:
                timer_id = self.interactor.CreateOneShotTimer(dt)
            else:
                timer_id = self.interactor.CreateRepeatingTimer(dt)
            return timer_id

        elif action in ("destroy", "stop"):
            if timer_id is not None:
                self.interactor.DestroyTimer(timer_id)
            else:
                vedo.logger.warning("please set a timer_id. Cannot stop timer.")
        else:
            e = f"in timer_callback(). Cannot understand action: {action}\n"
            e += " allowed actions are: ['start', 'stop']. Skipped."
            vedo.logger.error(e)
        return timer_id

    def add_observer(self, event_name, func, priority=0):
        """
        Add a callback function that will be called when an event occurs.
        Consider using `add_callback()` instead.
        """
        if not self.interactor:
            return self
        event_name = utils.get_vtk_name_event(event_name)
        idd = self.interactor.AddObserver(event_name, func, priority)
        return idd

    def compute_world_coordinate(
        self,
        pos2d,
        at=None,
        objs=(),
        bounds=(),
        offset=None,
        pixeltol=None,
        worldtol=None,
    ):
        """
        Transform a 2D point on the screen into a 3D point inside the rendering scene.
        If a set of meshes is passed then points are placed onto these.

        Arguments:
            pos2d : (list)
                2D screen coordinates point.
            at : (int)
                renderer number.
            objs : (list)
                list of Mesh objects to project the point onto.
            bounds : (list)
                specify a bounding box as [xmin,xmax, ymin,ymax, zmin,zmax].
            offset : (float)
                specify an offset value.
            pixeltol : (int)
                screen tolerance in pixels.
            worldtol : (float)
                world coordinates tolerance.

        Returns:
            numpy array, the point in 3D world coordinates.

        Examples:
            - [cut_freehand.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/cut_freehand.py)
            - [mousehover3.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mousehover3.py)

            ![](https://vedo.embl.es/images/basic/mousehover3.jpg)
        """
        if at is not None:
            renderer = self.renderers[at]
        else:
            renderer = self.renderer

        if not objs:
            pp = vtki.vtkFocalPlanePointPlacer()
        else:
            pp = vtki.vtkPolygonalSurfacePointPlacer()
            for ob in objs:
                pp.AddProp(ob.actor)

        if len(bounds) == 6:
            pp.SetPointBounds(bounds)
        if pixeltol:
            pp.SetPixelTolerance(pixeltol)
        if worldtol:
            pp.SetWorldTolerance(worldtol)
        if offset:
            pp.SetOffset(offset)

        worldPos = [0, 0, 0]
        worldOrient = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        pp.ComputeWorldPosition(renderer, pos2d, worldPos, worldOrient)
        # validw = pp.ValidateWorldPosition(worldPos, worldOrient)
        # validd = pp.ValidateDisplayPosition(renderer, pos2d)
        return np.array(worldPos)

    def compute_screen_coordinates(self, obj, full_window=False):
        """
        Given a 3D points in the current renderer (or full window),
        find the screen pixel coordinates.

        Example:
            ```python
            from vedo import *

            elli = Ellipsoid().point_size(5)

            plt = Plotter()
            plt.show(elli, "Press q to continue and print the info")

            xyscreen = plt.compute_screen_coordinates(elli)
            print('xyscreen coords:', xyscreen)

            # simulate an event happening at one point
            event = plt.fill_event(pos=xyscreen[123])
            print(event)
            ```
        """
        try:
            obj = obj.vertices
        except AttributeError:
            pass

        if utils.is_sequence(obj):
            pts = obj
        p2d = []
        cs = vtki.vtkCoordinate()
        cs.SetCoordinateSystemToWorld()
        cs.SetViewport(self.renderer)
        for p in pts:
            cs.SetValue(p)
            if full_window:
                p2d.append(cs.GetComputedDisplayValue(self.renderer))
            else:
                p2d.append(cs.GetComputedViewportValue(self.renderer))
        return np.array(p2d, dtype=int)

    def pick_area(self, pos1, pos2, at=None):
        """
        Pick all objects within a box defined by two corner points in 2D screen coordinates.

        Returns a frustum Mesh that contains the visible field of view.
        This can be used to select objects in a scene or select vertices.

        Example:
            ```python
            from vedo import *

            settings.enable_default_mouse_callbacks = False

            def mode_select(objs):
                print("Selected objects:", objs)
                d0 = mode.start_x, mode.start_y # display coords
                d1 = mode.end_x, mode.end_y

                frustum = plt.pick_area(d0, d1)
                col = np.random.randint(0, 10)
                infru = frustum.inside_points(mesh)
                infru.point_size(10).color(col)
                plt.add(frustum, infru).render()

            mesh = Mesh(dataurl+"cow.vtk")
            mesh.color("k5").linewidth(1)

            mode = interactor_modes.BlenderStyle()
            mode.callback_select = mode_select

            plt = Plotter().user_mode(mode)
            plt.show(mesh, axes=1)
            ```
        """
        if at is not None:
            ren = self.renderers[at]
        else:
            ren = self.renderer
        area_picker = vtki.vtkAreaPicker()
        area_picker.AreaPick(pos1[0], pos1[1], pos2[0], pos2[1], ren)
        planes = area_picker.GetFrustum()

        fru = vtki.new("FrustumSource")
        fru.SetPlanes(planes)
        fru.ShowLinesOff()
        fru.Update()

        afru = vedo.Mesh(fru.GetOutput())
        afru.alpha(0.1).lw(1).pickable(False)
        afru.name = "Frustum"
        return afru

    def _scan_input_return_acts(self, objs):
        # scan the input and return a list of actors
        if not utils.is_sequence(objs):
            objs = [objs]

        #################
        wannabe_acts2 = []
        for a in objs:

            try:
                wannabe_acts2.append(a.actor)
            except AttributeError:
                wannabe_acts2.append(a)  # already actor

            try:
                wannabe_acts2.append(a.scalarbar)
            except AttributeError:
                pass

            try:
                for sh in a.shadows:
                    wannabe_acts2.append(sh.actor)
            except AttributeError:
                pass

            try:
                wannabe_acts2.append(a.trail.actor)
                if a.trail.shadows:  # trails may also have shadows
                    for sh in a.trail.shadows:
                        wannabe_acts2.append(sh.actor)
            except AttributeError:
                pass

        #################
        scanned_acts = []
        for a in wannabe_acts2:  # scan content of list

            if a is None:
                pass

            elif isinstance(a, (vtki.vtkActor, vtki.vtkActor2D)):
                scanned_acts.append(a)

            elif isinstance(a, str):
                # assume a 2D comment was given
                changed = False  # check if one already exists so to just update text
                if self.renderer:  # might be jupyter
                    acs = self.renderer.GetActors2D()
                    acs.InitTraversal()
                    for i in range(acs.GetNumberOfItems()):
                        act = acs.GetNextItem()
                        if isinstance(act, vedo.shapes.Text2D):
                            aposx, aposy = act.GetPosition()
                            if aposx < 0.01 and aposy > 0.99:  # "top-left"
                                act.text(a)  # update content! no appending nada
                                changed = True
                                break
                    if not changed:
                        out = vedo.shapes.Text2D(a)  # append a new one
                        scanned_acts.append(out)
                # scanned_acts.append(vedo.shapes.Text2D(a)) # naive version

            elif isinstance(a, vtki.vtkPolyData):
                scanned_acts.append(vedo.Mesh(a).actor)

            elif isinstance(a, vtki.vtkImageData):
                scanned_acts.append(vedo.Volume(a).actor)

            elif isinstance(a, vedo.RectilinearGrid):
                scanned_acts.append(a.actor)

            elif isinstance(a, vedo.StructuredGrid):
                scanned_acts.append(a.actor)

            elif isinstance(a, vtki.vtkLight):
                self.renderer.AddLight(a)

            elif isinstance(a, vedo.visual.LightKit):
                a.lightkit.AddLightsToRenderer(self.renderer)

            elif isinstance(a, vtki.get_class("MultiBlockDataSet")):
                for i in range(a.GetNumberOfBlocks()):
                    b = a.GetBlock(i)
                    if isinstance(b, vtki.vtkPolyData):
                        scanned_acts.append(vedo.Mesh(b).actor)
                    elif isinstance(b, vtki.vtkImageData):
                        scanned_acts.append(vedo.Volume(b).actor)

            elif isinstance(a, (vtki.vtkProp, vtki.vtkInteractorObserver)):
                scanned_acts.append(a)

            elif "trimesh" in str(type(a)):
                scanned_acts.append(utils.trimesh2vedo(a))

            elif "meshlab" in str(type(a)):
                if "MeshSet" in str(type(a)):
                    for i in range(a.number_meshes()):
                        if a.mesh_id_exists(i):
                            scanned_acts.append(utils.meshlab2vedo(a.mesh(i)))
                else:
                    scanned_acts.append(utils.meshlab2vedo(a))

            elif "dolfin" in str(type(a)):  # assume a dolfin.Mesh object
                import vedo.dolfin as vdlf

                scanned_acts.append(vdlf.MeshActor(a).actor)

            elif "madcad" in str(type(a)):
                scanned_acts.append(utils.madcad2vedo(a).actor)

            elif "TetgenIO" in str(type(a)):
                scanned_acts.append(vedo.TetMesh(a).shrink(0.9).c("pink7").actor)

            elif "matplotlib.figure.Figure" in str(type(a)):
                scanned_acts.append(vedo.Image(a).clone2d("top-right", 0.6))

            else:
                vedo.logger.error(f"cannot understand input in show(): {type(a)}")

        return scanned_acts

    def show(
        self,
        *objects,
        at=None,
        axes=None,
        resetcam=None,
        zoom=False,
        interactive=None,
        viewup="",
        azimuth=0.0,
        elevation=0.0,
        roll=0.0,
        camera=None,
        mode=None,
        rate=None,
        bg=None,
        bg2=None,
        size=None,
        title=None,
        screenshot="",
    ):
        """
        Render a list of objects.

        Arguments:
            at : (int)
                number of the renderer to plot to, in case of more than one exists

            axes : (int)
                axis type-1 can be fully customized by passing a dictionary.
                Check `addons.Axes()` for the full list of options.
                set the type of axes to be shown:
                - 0,  no axes
                - 1,  draw three gray grid walls
                - 2,  show cartesian axes from (0,0,0)
                - 3,  show positive range of cartesian axes from (0,0,0)
                - 4,  show a triad at bottom left
                - 5,  show a cube at bottom left
                - 6,  mark the corners of the bounding box
                - 7,  draw a 3D ruler at each side of the cartesian axes
                - 8,  show the `vtkCubeAxesActor` object
                - 9,  show the bounding box outLine
                - 10, show three circles representing the maximum bounding box
                - 11, show a large grid on the x-y plane
                - 12, show polar axes
                - 13, draw a simple ruler at the bottom of the window

            azimuth/elevation/roll : (float)
                move camera accordingly the specified value

            viewup: str, list
                either `['x', 'y', 'z']` or a vector to set vertical direction

            resetcam : (bool)
                re-adjust camera position to fit objects

            camera : (dict, vtkCamera)
                camera parameters can further be specified with a dictionary
                assigned to the `camera` keyword (E.g. `show(camera={'pos':(1,2,3), 'thickness':1000,})`):
                - pos, `(list)`,  the position of the camera in world coordinates
                - focal_point `(list)`, the focal point of the camera in world coordinates
                - viewup `(list)`, the view up direction for the camera
                - distance `(float)`, set the focal point to the specified distance from the camera position.
                - clipping_range `(float)`, distance of the near and far clipping planes along the direction of projection.
                - parallel_scale `(float)`, scaling used for a parallel projection, i.e. the height of the viewport
                in world-coordinate distances. The default is 1.
                Note that the "scale" parameter works as an "inverse scale", larger numbers produce smaller images.
                This method has no effect in perspective projection mode.

                - thickness `(float)`, set the distance between clipping planes. This method adjusts the far clipping
                plane to be set a distance 'thickness' beyond the near clipping plane.

                - view_angle `(float)`, the camera view angle, which is the angular height of the camera view
                measured in degrees. The default angle is 30 degrees.
                This method has no effect in parallel projection mode.
                The formula for setting the angle up for perfect perspective viewing is:
                angle = 2*atan((h/2)/d) where h is the height of the RenderWindow
                (measured by holding a ruler up to your screen) and d is the distance from your eyes to the screen.

            interactive : (bool)
                pause and interact with window (True) or continue execution (False)

            rate : (float)
                maximum rate of `show()` in Hertz

            mode : (int, str)
                set the type of interaction:
                - 0 = TrackballCamera [default]
                - 1 = TrackballActor
                - 2 = JoystickCamera
                - 3 = JoystickActor
                - 4 = Flight
                - 5 = RubberBand2D
                - 6 = RubberBand3D
                - 7 = RubberBandZoom
                - 8 = Terrain
                - 9 = Unicam
                - 10 = Image
                - Check out `vedo.interaction_modes` for more options.

            bg : (str, list)
                background color in RGB format, or string name

            bg2 : (str, list)
                second background color to create a gradient background

            size : (str, list)
                size of the window, e.g. size="fullscreen", or size=[600,400]

            title : (str)
                window title text

            screenshot : (str)
                save a screenshot of the window to file
        """

        if settings.dry_run_mode >= 2:
            return self

        if self.wx_widget:
            return self

        if self.renderers:  # in case of notebooks

            if at is None:
                at = self.renderers.index(self.renderer)

            else:

                if at >= len(self.renderers):
                    t = f"trying to show(at={at}) but only {len(self.renderers)} renderers exist"
                    vedo.logger.error(t)
                    return self

                self.renderer = self.renderers[at]

        if title is not None:
            self.title = title

        if size is not None:
            self.size = size
            if self.size[0] == "f":  # full screen
                self.size = "fullscreen"
                self.window.SetFullScreen(True)
                self.window.BordersOn()
            else:
                self.window.SetSize(int(self.size[0]), int(self.size[1]))

        if settings.default_backend == "vtk":
            if str(bg).endswith(".hdr"):
                self._add_skybox(bg)
            else:
                if bg is not None:
                    self.backgrcol = vedo.get_color(bg)
                    self.renderer.SetBackground(self.backgrcol)
                if bg2 is not None:
                    self.renderer.GradientBackgroundOn()
                    self.renderer.SetBackground2(vedo.get_color(bg2))

        if axes is not None:
            if isinstance(axes, vedo.Assembly):  # user passing show(..., axes=myaxes)
                objects = list(objects)
                objects.append(axes)  # move it into the list of normal things to show
                axes = 0
            self.axes = axes

        if interactive is not None:
            self._interactive = interactive
        if self.offscreen:
            self._interactive = False

        # camera stuff
        if resetcam is not None:
            self.resetcam = resetcam

        if camera is not None:
            self.resetcam = False
            viewup = ""
            if isinstance(camera, vtki.vtkCamera):
                cameracopy = vtki.vtkCamera()
                cameracopy.DeepCopy(camera)
                self.camera = cameracopy
            else:
                self.camera = utils.camera_from_dict(camera)

        self.add(objects)

        # Backend ###############################################################
        if settings.default_backend in ["k3d"]:
            return backends.get_notebook_backend(self.objects)
        #########################################################################

        for ia in utils.flatten(objects):
            try:
                # fix gray color labels and title to white or black
                ltc = np.array(ia.scalarbar.GetLabelTextProperty().GetColor())
                if np.linalg.norm(ltc - (0.5, 0.5, 0.5)) / 3 < 0.05:
                    c = (0.9, 0.9, 0.9)
                    if np.sum(self.renderer.GetBackground()) > 1.5:
                        c = (0.1, 0.1, 0.1)
                    ia.scalarbar.GetLabelTextProperty().SetColor(c)
                    ia.scalarbar.GetTitleTextProperty().SetColor(c)
            except AttributeError:
                pass

        if self.sharecam:
            for r in self.renderers:
                r.SetActiveCamera(self.camera)

        if self.axes is not None:
            if viewup != "2d" or self.axes in [1, 8] or isinstance(self.axes, dict):
                bns = self.renderer.ComputeVisiblePropBounds()
                addons.add_global_axes(self.axes, bounds=bns)

        # Backend ###############################################################
        if settings.default_backend in ["ipyvtk", "trame"]:
            return backends.get_notebook_backend()
        #########################################################################

        if self.resetcam:
            self.renderer.ResetCamera()

        if len(self.renderers) > 1:
            self.add_renderer_frame()

        if settings.default_backend == "2d" and not zoom:
            zoom = "tightest"

        if zoom:
            if zoom == "tight":
                self.reset_camera(tight=0.04)
            elif zoom == "tightest":
                self.reset_camera(tight=0.0001)
            else:
                self.camera.Zoom(zoom)
        if elevation:
            self.camera.Elevation(elevation)
        if azimuth:
            self.camera.Azimuth(azimuth)
        if roll:
            self.camera.Roll(roll)

        if len(viewup) > 0:
            b = self.renderer.ComputeVisiblePropBounds()
            cm = np.array([(b[1] + b[0])/2, (b[3] + b[2])/2, (b[5] + b[4])/2])
            sz = np.array([(b[1] - b[0]), (b[3] - b[2]), (b[5] - b[4])])
            if viewup == "x":
                sz = np.linalg.norm(sz)
                self.camera.SetViewUp([1, 0, 0])
                self.camera.SetPosition(cm + sz)
            elif viewup == "y":
                sz = np.linalg.norm(sz)
                self.camera.SetViewUp([0, 1, 0])
                self.camera.SetPosition(cm + sz)
            elif viewup == "z":
                sz = np.array([(b[1]-b[0])*0.7, -(b[3]-b[2])*1.0, (b[5]-b[4])*1.2])
                self.camera.SetViewUp([0, 0, 1])
                self.camera.SetPosition(cm + 2 * sz)
            elif utils.is_sequence(viewup):
                sz = np.linalg.norm(sz)
                self.camera.SetViewUp(viewup)
                cpos = np.cross([0, 1, 0], viewup)
                self.camera.SetPosition(cm - 2 * sz * cpos)

        self.renderer.ResetCameraClippingRange()

        self.initialize_interactor()

        if settings.immediate_rendering:
            self.window.Render()  ##################### <-------------- Render

        if self.interactor:  # can be offscreen or not the vtk backend..

            self.window.SetWindowName(self.title)

            # pic = vedo.Image(vedo.dataurl+'images/vtk_logo.png')
            # pic = vedo.Image('/home/musy/Downloads/icons8-3d-96.png')
            # print(pic.dataset)# Array 0 name PNGImage
            # self.window.SetIcon(pic.dataset)

            try:
                # Needs "pip install pyobjc" on Mac OSX
                if (
                    self._cocoa_initialized is False
                    and "Darwin" in vedo.sys_platform
                    and not self.offscreen
                ):
                    self._cocoa_initialized = True
                    from Cocoa import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
                    pid = os.getpid()
                    x = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(pid))
                    x.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            except:
                # vedo.logger.debug("On Mac OSX try: pip install pyobjc")
                pass

            # Set the interaction style
            if mode is not None:
                self.user_mode(mode)
            if self.qt_widget and mode is None:
                self.user_mode(0)

            if screenshot:
                self.screenshot(screenshot)

            if self._interactive:
                self.interactor.Start()
                if self._must_close_now:
                    self.interactor.GetRenderWindow().Finalize()
                    self.interactor.TerminateApp()
                    self.camera = None
                    self.renderer = None
                    self.renderers = []
                    self.window = None
                    self.interactor = None
                return self

            if rate:
                if self.clock is None:  # set clock and limit rate
                    self._clockt0 = time.time()
                    self.clock = 0.0
                else:
                    t = time.time() - self._clockt0
                    elapsed = t - self.clock
                    mint = 1.0 / rate
                    if elapsed < mint:
                        time.sleep(mint - elapsed)
                    self.clock = time.time() - self._clockt0

        # 2d ####################################################################
        if settings.default_backend == "2d":
            return backends.get_notebook_backend()
        #########################################################################

        return self


    def add_inset(self, *objects, **options):
        """Add a draggable inset space into a renderer.

        Arguments:
            at : (int)
                specify the renderer number
            pos : (list)
                icon position in the range [1-4] indicating one of the 4 corners,
                or it can be a tuple (x,y) as a fraction of the renderer size.
            size : (float)
                size of the square inset
            draggable : (bool)
                if True the subrenderer space can be dragged around
            c : (color)
                color of the inset frame when dragged

        Examples:
            - [inset.py](https://github.com/marcomusy/vedo/tree/master/examples/other/inset.py)

            ![](https://user-images.githubusercontent.com/32848391/56758560-3c3f1300-6797-11e9-9b33-49f5a4876039.jpg)
        """
        if not self.interactor:
            return None

        if not self.renderer:
            vedo.logger.warning("call add_inset() only after first rendering of the scene.")
            return None

        options = dict(options)
        pos = options.pop("pos", 0)
        size = options.pop("size", 0.1)
        c = options.pop("c", "lb")
        at = options.pop("at", None)
        draggable = options.pop("draggable", True)

        widget = vtki.vtkOrientationMarkerWidget()
        r, g, b = vedo.get_color(c)
        widget.SetOutlineColor(r, g, b)
        if len(objects) == 1:
            widget.SetOrientationMarker(objects[0].actor)
        else:
            widget.SetOrientationMarker(vedo.Assembly(objects))

        widget.SetInteractor(self.interactor)

        if utils.is_sequence(pos):
            widget.SetViewport(pos[0] - size, pos[1] - size, pos[0] + size, pos[1] + size)
        else:
            if pos < 2:
                widget.SetViewport(0, 1 - 2 * size, size * 2, 1)
            elif pos == 2:
                widget.SetViewport(1 - 2 * size, 1 - 2 * size, 1, 1)
            elif pos == 3:
                widget.SetViewport(0, 0, size * 2, size * 2)
            elif pos == 4:
                widget.SetViewport(1 - 2 * size, 0, 1, size * 2)
        widget.EnabledOn()
        widget.SetInteractive(draggable)
        if at is not None and at < len(self.renderers):
            widget.SetCurrentRenderer(self.renderers[at])
        else:
            widget.SetCurrentRenderer(self.renderer)
        self.widgets.append(widget)
        return widget

    def clear(self, at=None, deep=False):
        """Clear the scene from all meshes and volumes."""
        if at is not None:
            renderer = self.renderers[at]
        else:
            renderer = self.renderer
        if not renderer:
            return self

        if deep:
            renderer.RemoveAllViewProps()
        else:
            for ob in set(
                self.get_meshes()
                + self.get_volumes()
                + self.objects
                + self.axes_instances
            ):
                if isinstance(ob, vedo.shapes.Text2D):
                    continue
                self.remove(ob)
                try:
                    if ob.scalarbar:
                        self.remove(ob.scalarbar)
                except AttributeError:
                    pass
        return self

    def break_interaction(self):
        """Break window interaction and return to the python execution flow"""
        if self.interactor:
            self.check_actors_trasform()
            self.interactor.ExitCallback()
        return self

    def user_mode(self, mode):
        """
        Modify the user interaction mode.

        Examples:
            ```python
            from vedo import *
            mode = interactor_modes.MousePan()
            mesh = Mesh(dataurl+"cow.vtk")
            plt = Plotter().user_mode(mode)
            plt.show(mesh, axes=1)
           ```
        See also:
        [VTK interactor styles](https://vtk.org/doc/nightly/html/classvtkInteractorStyle.html)
        """
        if not self.interactor:
            return None
        
        curr_style = self.interactor.GetInteractorStyle().GetClassName()
        # print("Current style:", curr_style)
        if curr_style.endswith("Actor"):
            self.check_actors_trasform()

        if isinstance(mode, (str, int)):
            # Set the style of interaction
            # see https://vtk.org/doc/nightly/html/classvtkInteractorStyle.html
            if   mode in (0, "TrackballCamera"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleTrackballCamera"))
                self.interactor.RemoveObservers("CharEvent")
            elif mode in (1, "TrackballActor"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleTrackballActor"))
            elif mode in (2, "JoystickCamera"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleJoystickCamera"))
            elif mode in (3, "JoystickActor"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleJoystickActor"))
            elif mode in (4, "Flight"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleFlight"))
            elif mode in (5, "RubberBand2D"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleRubberBand2D"))
            elif mode in (6, "RubberBand3D"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleRubberBand3D"))
            elif mode in (7, "RubberBandZoom"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleRubberBandZoom"))
            elif mode in (8, "Terrain"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleTerrain"))
            elif mode in (9, "Unicam"):
                self.interactor.SetInteractorStyle(vtki.new("InteractorStyleUnicam"))
            elif mode in (10, "Image", "image", "2d"):
                astyle = vtki.new("InteractorStyleImage")
                astyle.SetInteractionModeToImage3D()
                self.interactor.SetInteractorStyle(astyle)
            else:
                vedo.logger.warning(f"Unknown interaction mode: {mode}")

        elif isinstance(mode, vtki.vtkInteractorStyleUser):
            # set a custom interactor style
            mode.interactor = self.interactor
            mode.renderer = self.renderer
            mode.SetInteractor(self.interactor)
            mode.SetDefaultRenderer(self.renderer)
            self.interactor.SetInteractorStyle(mode)

        return self

    def close(self):
        """Close the plotter."""
        # https://examples.vtk.org/site/Cxx/Visualization/CloseWindow/
        vedo.last_figure = None
        self.last_event = None
        self.sliders = []
        self.buttons = []
        self.widgets = []
        self.hover_legends = []
        self.background_renderer = None
        self._extralight = None

        self.hint_widget = None
        self.cutter_widget = None

        if settings.dry_run_mode >= 2:
            return self
        
        if not hasattr(self, "window"):
            return self
        if not self.window:
            return self
        if not hasattr(self, "interactor"):
            return self
        if not self.interactor:
            return self

        ###################################################
        try:
            if "Darwin" in vedo.sys_platform:
                self.interactor.ProcessEvents()
        except:
            pass

        self._must_close_now = True

        if vedo.plotter_instance == self:
            vedo.plotter_instance = None

        if self.interactor and self._interactive:
            self.break_interaction()
        elif self._must_close_now:
            # dont call ExitCallback here
            self.interactor.GetRenderWindow().Finalize()
            self.interactor.TerminateApp()
            self.camera = None
            self.renderer = None
            self.renderers = []
            self.window = None
            self.interactor = None
        return self

    # def close(self):
    #     """Close the Plotter instance and release resources."""
    #     self.close_window()
    #     if vedo.plotter_instance == self:
    #         vedo.plotter_instance = None

    @property
    def camera(self):
        """Return the current active camera."""
        if self.renderer:
            return self.renderer.GetActiveCamera()

    @camera.setter
    def camera(self, cam):
        if self.renderer:
            if isinstance(cam, dict):
                cam = utils.camera_from_dict(cam)
            self.renderer.SetActiveCamera(cam)

    def screenshot(self, filename="screenshot.png", scale=1, asarray=False):
        """
        Take a screenshot of the Plotter window.

        Arguments:
            scale : (int)
                set image magnification as an integer multiplicating factor
            asarray : (bool)
                return a numpy array of the image instead of writing a file

        Warning:
            If you get black screenshots try to set `interactive=False` in `show()`
            then call `screenshot()` and `plt.interactive()` afterwards.

        Example:
            ```py
            from vedo import *
            sphere = Sphere().linewidth(1)
            plt = show(sphere, interactive=False)
            plt.screenshot('image.png')
            plt.interactive()
            plt.close()
            ```

        Example:
            ```py
            from vedo import *
            sphere = Sphere().linewidth(1)
            plt = show(sphere, interactive=False)
            plt.screenshot('anotherimage.png')
            plt.interactive()
            plt.close()
            ```
        """
        return vedo.file_io.screenshot(filename, scale, asarray)

    def toimage(self, scale=1):
        """
        Generate a `Image` object from the current rendering window.

        Arguments:
            scale : (int)
                set image magnification as an integer multiplicating factor
        """
        if settings.screeshot_large_image:
            w2if = vtki.new("RenderLargeImage")
            w2if.SetInput(self.renderer)
            w2if.SetMagnification(scale)
        else:
            w2if = vtki.new("WindowToImageFilter")
            w2if.SetInput(self.window)
            if hasattr(w2if, "SetScale"):
                w2if.SetScale(scale, scale)
            if settings.screenshot_transparent_background:
                w2if.SetInputBufferTypeToRGBA()
            w2if.ReadFrontBufferOff()  # read from the back buffer
        w2if.Update()
        return vedo.image.Image(w2if.GetOutput())

    def export(self, filename="scene.npz", binary=False):
        """
        Export scene to file to HTML, X3D or Numpy file.

        Examples:
            - [export_x3d.py](https://github.com/marcomusy/vedo/tree/master/examples/other/export_x3d.py)
            - [export_numpy.py](https://github.com/marcomusy/vedo/tree/master/examples/other/export_numpy.py)
        """
        vedo.file_io.export_window(filename, binary=binary)
        return self

    def color_picker(self, xy, verbose=False):
        """Pick color of specific (x,y) pixel on the screen."""
        w2if = vtki.new("WindowToImageFilter")
        w2if.SetInput(self.window)
        w2if.ReadFrontBufferOff()
        w2if.Update()
        nx, ny = self.window.GetSize()
        varr = w2if.GetOutput().GetPointData().GetScalars()

        arr = utils.vtk2numpy(varr).reshape(ny, nx, 3)
        x, y = int(xy[0]), int(xy[1])
        if y < ny and x < nx:

            rgb = arr[y, x]

            if verbose:
                vedo.printc(":rainbow:Pixel", [x, y], "has RGB[", end="")
                vedo.printc("█", c=[rgb[0], 0, 0], end="")
                vedo.printc("█", c=[0, rgb[1], 0], end="")
                vedo.printc("█", c=[0, 0, rgb[2]], end="")
                vedo.printc("] = ", end="")
                cnm = vedo.get_color_name(rgb)
                if np.sum(rgb) < 150:
                    vedo.printc(
                        rgb.tolist(),
                        vedo.colors.rgb2hex(np.array(rgb) / 255),
                        c="w",
                        bc=rgb,
                        invert=1,
                        end="",
                    )
                    vedo.printc("  -> " + cnm, invert=1, c="w")
                else:
                    vedo.printc(
                        rgb.tolist(),
                        vedo.colors.rgb2hex(np.array(rgb) / 255),
                        c=rgb,
                        end="",
                    )
                    vedo.printc("  -> " + cnm, c=cnm)

            return rgb

        return None

    #######################################################################
    def _default_mouseleftclick(self, iren, event):
        x, y = iren.GetEventPosition()
        renderer = iren.FindPokedRenderer(x, y)
        picker = vtki.vtkPropPicker()
        picker.PickProp(x, y, renderer)

        self.renderer = renderer

        clicked_actor = picker.GetActor()
        # clicked_actor2D = picker.GetActor2D()

        # print('_default_mouseleftclick mouse at', x, y)
        # print("picked Volume:",   [picker.GetVolume()])
        # print("picked Actor2D:",  [picker.GetActor2D()])
        # print("picked Assembly:", [picker.GetAssembly()])
        # print("picked Prop3D:",   [picker.GetProp3D()])

        if not clicked_actor:
            clicked_actor = picker.GetAssembly()

        if not clicked_actor:
            clicked_actor = picker.GetProp3D()

        if not hasattr(clicked_actor, "GetPickable") or not clicked_actor.GetPickable():
            return

        self.picked3d = picker.GetPickPosition()
        self.picked2d = np.array([x, y])

        if not clicked_actor:
            return

        self.justremoved = None
        self.clicked_actor = clicked_actor

        try:  # might not be a vedo obj
            self.clicked_object = clicked_actor.retrieve_object()
            # save this info in the object itself
            self.clicked_object.picked3d = self.picked3d
            self.clicked_object.picked2d = self.picked2d
        except AttributeError:
            pass

        # -----------
        if "Histogram1D" in picker.GetAssembly().__class__.__name__:
            histo = picker.GetAssembly()
            if histo.verbose:
                x = self.picked3d[0]
                idx = np.digitize(x, histo.edges) - 1
                f = histo.frequencies[idx]
                cn = histo.centers[idx]
                vedo.colors.printc(f"{histo.name}, bin={idx}, center={cn}, value={f}")

    #######################################################################
    def _default_keypress(self, iren, event):
        # NB: qt creates and passes a vtkGenericRenderWindowInteractor

        key = iren.GetKeySym()

        if "_L" in key or "_R" in key:
            return

        if iren.GetShiftKey():
            key = key.upper()

        if iren.GetControlKey():
            key = "Ctrl+" + key

        if iren.GetAltKey():
            key = "Alt+" + key

        #######################################################
        # utils.vedo.printc('Pressed key:', key, c='y', box='-')
        # print(key, iren.GetShiftKey(), iren.GetAltKey(), iren.GetControlKey(),
        #       iren.GetKeyCode(), iren.GetRepeatCount())
        #######################################################

        x, y = iren.GetEventPosition()
        renderer = iren.FindPokedRenderer(x, y)

        if key in ["q", "Return"]:
            self.break_interaction()
            return

        elif key in ["Ctrl+q", "Ctrl+w", "Escape"]:
            self.close()
            return

        elif key == "F1":
            vedo.logger.info("Execution aborted. Exiting python kernel now.")
            self.break_interaction()
            sys.exit(0)

        elif key == "Down":
            if self.clicked_object and self.clicked_object in self.get_meshes():
                self.clicked_object.alpha(0.02)
                if hasattr(self.clicked_object, "properties_backface"):
                    bfp = self.clicked_actor.GetBackfaceProperty()
                    self.clicked_object.properties_backface = bfp  # save it
                    self.clicked_actor.SetBackfaceProperty(None)
            else:
                for obj in self.get_meshes():
                    if obj:
                        obj.alpha(0.02)
                        bfp = obj.actor.GetBackfaceProperty()
                        if bfp and hasattr(obj, "properties_backface"):
                            obj.properties_backface = bfp
                            obj.actor.SetBackfaceProperty(None)

        elif key == "Left":
            if self.clicked_object and self.clicked_object in self.get_meshes():
                ap = self.clicked_object.properties
                aal = max([ap.GetOpacity() * 0.75, 0.01])
                ap.SetOpacity(aal)
                bfp = self.clicked_actor.GetBackfaceProperty()
                if bfp and hasattr(self.clicked_object, "properties_backface"):
                    self.clicked_object.properties_backface = bfp
                    self.clicked_actor.SetBackfaceProperty(None)
            else:
                for a in self.get_meshes():
                    if a:
                        ap = a.properties
                        aal = max([ap.GetOpacity() * 0.75, 0.01])
                        ap.SetOpacity(aal)
                        bfp = a.actor.GetBackfaceProperty()
                        if bfp and hasattr(a, "properties_backface"):
                            a.properties_backface = bfp
                            a.actor.SetBackfaceProperty(None)

        elif key == "Right":
            if self.clicked_object and self.clicked_object in self.get_meshes():
                ap = self.clicked_object.properties
                aal = min([ap.GetOpacity() * 1.25, 1.0])
                ap.SetOpacity(aal)
                if (
                    aal == 1
                    and hasattr(self.clicked_object, "properties_backface")
                    and self.clicked_object.properties_backface
                ):
                    # put back
                    self.clicked_actor.SetBackfaceProperty(
                        self.clicked_object.properties_backface)
            else:
                for a in self.get_meshes():
                    if a:
                        ap = a.properties
                        aal = min([ap.GetOpacity() * 1.25, 1.0])
                        ap.SetOpacity(aal)
                        if aal == 1 and hasattr(a, "properties_backface") and a.properties_backface:
                            a.actor.SetBackfaceProperty(a.properties_backface)

        elif key == "Up":
            if self.clicked_object and self.clicked_object in self.get_meshes():
                self.clicked_object.properties.SetOpacity(1)
                if hasattr(self.clicked_object, "properties_backface") and self.clicked_object.properties_backface:
                    self.clicked_object.actor.SetBackfaceProperty(self.clicked_object.properties_backface)
            else:
                for a in self.get_meshes():
                    if a:
                        a.properties.SetOpacity(1)
                        if hasattr(a, "properties_backface") and a.properties_backface:
                            a.actor.SetBackfaceProperty(a.properties_backface)

        elif key == "P":
            if self.clicked_object and self.clicked_object in self.get_meshes():
                objs = [self.clicked_object]
            else:
                objs = self.get_meshes()
            for ia in objs:
                try:
                    ps = ia.properties.GetPointSize()
                    if ps > 1:
                        ia.properties.SetPointSize(ps - 1)
                    ia.properties.SetRepresentationToPoints()
                except AttributeError:
                    pass

        elif key == "p":
            if self.clicked_object and self.clicked_object in self.get_meshes():
                objs = [self.clicked_object]
            else:
                objs = self.get_meshes()
            for ia in objs:
                try:
                    ps = ia.properties.GetPointSize()
                    ia.properties.SetPointSize(ps + 2)
                    ia.properties.SetRepresentationToPoints()
                except AttributeError:
                    pass

        elif key == "U":
            pval = renderer.GetActiveCamera().GetParallelProjection()
            renderer.GetActiveCamera().SetParallelProjection(not pval)
            if pval:
                renderer.ResetCamera()

        elif key == "r":
            renderer.ResetCamera()

        elif key == "h":
            msg  = f" vedo {vedo.__version__}"
            msg += f" | vtk {vtki.vtkVersion().GetVTKVersion()}"
            msg += f" | numpy {np.__version__}"
            msg += f" | python {sys.version_info[0]}.{sys.version_info[1]}, press: "
            vedo.printc(msg.ljust(75), invert=True)
            msg = (
                "    i     print info about the last clicked object     \n"
                "    I     print color of the pixel under the mouse     \n"
                "    Y     show the pipeline for this object as a graph \n"
                "    <- -> use arrows to reduce/increase opacity        \n"
                "    x     toggle mesh visibility                       \n"
                "    w     toggle wireframe/surface style               \n"
                "    l     toggle surface edges visibility              \n"
                "    p/P   hide surface faces and show only points      \n"
                "    1-3   cycle surface color (2=light, 3=dark)        \n"
                "    4     cycle color map (press shift-4 to go back)   \n"
                "    5-6   cycle point-cell arrays (shift to go back)   \n"
                "    7-8   cycle background and gradient color          \n"
                "    09+-  cycle axes styles (on keypad, or press +/-)  \n"
                "    k     cycle available lighting styles              \n"
                "    K     toggle shading as flat or phong              \n"
                "    A     toggle anti-aliasing                         \n"
                "    D     toggle depth-peeling (for transparencies)    \n"
                "    U     toggle perspective/parallel projection       \n"
                "    o/O   toggle extra light to scene and rotate it    \n"
                "    a     toggle interaction to Actor Mode             \n"
                "    n     toggle surface normals                       \n"
                "    r     reset camera position                        \n"
                "    R     reset camera to the closest orthogonal view  \n"
                "    .     fly camera to the last clicked point         \n"
                "    C     print the current camera parameters state    \n"
                "    X     invoke a cutter widget tool                  \n"
                "    S     save a screenshot of the current scene       \n"
                "    E/F   export 3D scene to numpy file or X3D         \n"
                "    q     return control to python script              \n"
                "    Esc   abort execution and exit python kernel       "
            )
            vedo.printc(msg, dim=True, italic=True, bold=True)
            vedo.printc(
                " Check out the documentation at:  https://vedo.embl.es ".ljust(75),
                invert=True,
                bold=True,
            )
            return

        elif key == "a":
            cur = iren.GetInteractorStyle()
            if isinstance(cur, vtki.get_class("InteractorStyleTrackballCamera")):
                msg  = "Interactor style changed to TrackballActor\n"
                msg += "  you can now move and rotate individual meshes:\n"
                msg += "  press X twice to save the repositioned mesh\n"
                msg += "  press 'a' to go back to normal style"
                vedo.printc(msg)
                iren.SetInteractorStyle(vtki.new("InteractorStyleTrackballActor"))
            else:
                iren.SetInteractorStyle(vtki.new("InteractorStyleTrackballCamera"))
            return

        elif key == "A":  # toggle antialiasing
            msam = self.window.GetMultiSamples()
            if not msam:
                self.window.SetMultiSamples(16)
            else:
                self.window.SetMultiSamples(0)
            msam = self.window.GetMultiSamples()
            if msam:
                vedo.printc(f"Antialiasing set to {msam} samples", c=bool(msam))
            else:
                vedo.printc("Antialiasing disabled", c=bool(msam))

        elif key == "D":  # toggle depthpeeling
            udp = not renderer.GetUseDepthPeeling()
            renderer.SetUseDepthPeeling(udp)
            # self.renderer.SetUseDepthPeelingForVolumes(udp)
            if udp:
                self.window.SetAlphaBitPlanes(1)
                renderer.SetMaximumNumberOfPeels(settings.max_number_of_peels)
                renderer.SetOcclusionRatio(settings.occlusion_ratio)
            self.interactor.Render()
            wasUsed = renderer.GetLastRenderingUsedDepthPeeling()
            rnr = self.renderers.index(renderer)
            vedo.printc(f"Depth peeling set to {udp} for renderer nr.{rnr}", c=udp)
            if not wasUsed and udp:
                vedo.printc("\t...but last rendering did not actually used it!", c=udp, invert=True)
            return

        elif key == "period":
            if self.picked3d:
                self.fly_to(self.picked3d)
            return

        elif key == "S":
            vedo.file_io.screenshot("screenshot.png")
            vedo.printc(r":camera: Saved rendering window to 'screenshot.png'", c="b")
            return

        elif key == "C":
            # Precision needs to be 7 (or even larger) to guarantee a consistent camera when
            #   the model coordinates are not centered at (0, 0, 0) and the mode is large.
            # This could happen for plotting geological models with UTM coordinate systems
            cam = renderer.GetActiveCamera()
            vedo.printc("\n###################################################", c="y")
            vedo.printc("## Template python code to position this camera: ##", c="y")
            vedo.printc("cam = dict(", c="y")
            vedo.printc("    position=" + utils.precision(cam.GetPosition(), 6) + ",", c="y")
            vedo.printc("    focal_point=" + utils.precision(cam.GetFocalPoint(), 6) + ",", c="y")
            vedo.printc("    viewup=" + utils.precision(cam.GetViewUp(), 6) + ",", c="y")
            vedo.printc("    roll=" + utils.precision(cam.GetRoll(), 6) + ",", c="y")
            if cam.GetParallelProjection():
                vedo.printc('    parallel_scale='+utils.precision(cam.GetParallelScale(),6)+',', c='y')
            else:
                vedo.printc('    distance='     +utils.precision(cam.GetDistance(),6)+',', c='y')
            vedo.printc('    clipping_range='+utils.precision(cam.GetClippingRange(),6)+',', c='y')
            vedo.printc(')', c='y')
            vedo.printc('show(mymeshes, camera=cam)', c='y')
            vedo.printc('###################################################', c='y')
            return

        elif key == "R":
            self.reset_viewup()

        elif key == "w":
            try:
                if self.clicked_object.properties.GetRepresentation() == 1:  # toggle
                    self.clicked_object.properties.SetRepresentationToSurface()
                else:
                    self.clicked_object.properties.SetRepresentationToWireframe()
            except AttributeError:
                pass

        elif key == "1":
            try:
                self._icol += 1
                self.clicked_object.mapper.ScalarVisibilityOff()
                pal = vedo.colors.palettes[settings.palette % len(vedo.colors.palettes)]
                self.clicked_object.c(pal[(self._icol) % 10])
                self.remove(self.clicked_object.scalarbar)
            except AttributeError:
                pass

        elif key == "2": # dark colors
            try:
                bsc = ["k1", "k2", "k3", "k4",
                    "b1", "b2", "b3", "b4",
                    "p1", "p2", "p3", "p4",
                    "g1", "g2", "g3", "g4",
                    "r1", "r2", "r3", "r4",
                    "o1", "o2", "o3", "o4",
                    "y1", "y2", "y3", "y4"]
                self._icol += 1
                if self.clicked_object:
                    self.clicked_object.mapper.ScalarVisibilityOff()
                    newcol = vedo.get_color(bsc[(self._icol) % len(bsc)])
                    self.clicked_object.c(newcol)
                    self.remove(self.clicked_object.scalarbar)
            except AttributeError:
                pass

        elif key == "3": # light colors
            try:
                bsc = ["k6", "k7", "k8", "k9",
                    "b6", "b7", "b8", "b9",
                    "p6", "p7", "p8", "p9",
                    "g6", "g7", "g8", "g9",
                    "r6", "r7", "r8", "r9",
                    "o6", "o7", "o8", "o9",
                    "y6", "y7", "y8", "y9"]
                self._icol += 1
                if self.clicked_object:
                    self.clicked_object.mapper.ScalarVisibilityOff()
                    newcol = vedo.get_color(bsc[(self._icol) % len(bsc)])
                    self.clicked_object.c(newcol)
                    self.remove(self.clicked_object.scalarbar)
            except AttributeError:
                pass

        elif key == "4":  # cmap name cycle
            ob = self.clicked_object
            if not isinstance(ob, (vedo.Points, vedo.UnstructuredGrid)):
                return
            if not ob.mapper.GetScalarVisibility():
                return
            onwhat = ob.mapper.GetScalarModeAsString()  # UsePointData/UseCellData

            cmap_names = [
                "Accent", "Paired",
                "rainbow", "rainbow_r",
                "Spectral", "Spectral_r",
                "gist_ncar", "gist_ncar_r",
                "viridis", "viridis_r",
                "hot", "hot_r",
                "terrain", "ocean",
                "coolwarm", "seismic", "PuOr", "RdYlGn",
            ]
            try:
                i = cmap_names.index(ob._cmap_name)
                if iren.GetShiftKey():
                    i -= 1
                else:
                    i += 1
                if i >= len(cmap_names):
                    i = 0
                if i < 0:
                    i = len(cmap_names) - 1
            except ValueError:
                i = 0

            ob._cmap_name = cmap_names[i]
            ob.cmap(ob._cmap_name, on=onwhat)
            if ob.scalarbar:
                if isinstance(ob.scalarbar, vtki.vtkActor2D):
                    self.remove(ob.scalarbar)
                    title = ob.scalarbar.GetTitle()
                    ob.add_scalarbar(title=title)
                    self.add(ob.scalarbar).render()
                elif isinstance(ob.scalarbar, vedo.Assembly):
                    self.remove(ob.scalarbar)
                    ob.add_scalarbar3d(title=ob._cmap_name)
                    self.add(ob.scalarbar)

            vedo.printc(
                f"Name:'{ob.name}'," if ob.name else "",
                f"range:{utils.precision(ob.mapper.GetScalarRange(),3)},",
                f"colormap:'{ob._cmap_name}'", c="g", bold=False,
            )

        elif key == "5":  # cycle pointdata array
            ob = self.clicked_object
            if not isinstance(ob, (vedo.Points, vedo.UnstructuredGrid)):
                return

            arrnames = ob.pointdata.keys()
            arrnames = [a for a in arrnames if "normal" not in a.lower()]
            arrnames = [a for a in arrnames if "tcoord" not in a.lower()]
            arrnames = [a for a in arrnames if "textur" not in a.lower()]
            if len(arrnames) == 0:
                return
            ob.mapper.SetScalarVisibility(1)

            if not ob._cmap_name:
                ob._cmap_name = "rainbow"

            try:
                curr_name = ob.dataset.GetPointData().GetScalars().GetName()
                i = arrnames.index(curr_name)
                if "normals" in curr_name.lower():
                    return
                if iren.GetShiftKey():
                    i -= 1
                else:
                    i += 1
                if i >= len(arrnames):
                    i = 0
                if i < 0:
                    i = len(arrnames) - 1
            except (ValueError, AttributeError):
                i = 0

            ob.cmap(ob._cmap_name, arrnames[i], on="points")
            if ob.scalarbar:
                if isinstance(ob.scalarbar, vtki.vtkActor2D):
                    self.remove(ob.scalarbar)
                    title = ob.scalarbar.GetTitle()
                    ob.scalarbar = None
                    ob.add_scalarbar(title=arrnames[i])
                    self.add(ob.scalarbar)
                elif isinstance(ob.scalarbar, vedo.Assembly):
                    self.remove(ob.scalarbar)
                    ob.scalarbar = None
                    ob.add_scalarbar3d(title=arrnames[i])
                    self.add(ob.scalarbar)
            else:
                vedo.printc(
                    f"Name:'{ob.name}'," if ob.name else "",
                    f"active pointdata array: '{arrnames[i]}'",
                    c="g", bold=False,
                )

        elif key == "6":  # cycle celldata array
            ob = self.clicked_object
            if not isinstance(ob, (vedo.Points, vedo.UnstructuredGrid)):
                return

            arrnames = ob.celldata.keys()
            arrnames = [a for a in arrnames if "normal" not in a.lower()]
            arrnames = [a for a in arrnames if "tcoord" not in a.lower()]
            arrnames = [a for a in arrnames if "textur" not in a.lower()]
            if len(arrnames) == 0:
                return
            ob.mapper.SetScalarVisibility(1)

            if not ob._cmap_name:
                ob._cmap_name = "rainbow"

            try:
                curr_name = ob.dataset.GetCellData().GetScalars().GetName()
                i = arrnames.index(curr_name)
                if "normals" in curr_name.lower():
                    return
                if iren.GetShiftKey():
                    i -= 1
                else:
                    i += 1
                if i >= len(arrnames):
                    i = 0
                if i < 0:
                    i = len(arrnames) - 1
            except (ValueError, AttributeError):
                i = 0

            ob.cmap(ob._cmap_name, arrnames[i], on="cells")
            if ob.scalarbar:
                if isinstance(ob.scalarbar, vtki.vtkActor2D):
                    self.remove(ob.scalarbar)
                    title = ob.scalarbar.GetTitle()
                    ob.scalarbar = None
                    ob.add_scalarbar(title=arrnames[i])
                    self.add(ob.scalarbar)
                elif isinstance(ob.scalarbar, vedo.Assembly):
                    self.remove(ob.scalarbar)
                    ob.scalarbar = None
                    ob.add_scalarbar3d(title=arrnames[i])
                    self.add(ob.scalarbar)
            else:
                vedo.printc(
                    f"Name:'{ob.name}'," if ob.name else "",
                    f"active celldata array: '{arrnames[i]}'",
                    c="g", bold=False,
                )

        elif key == "7":
            bgc = np.array(renderer.GetBackground()).sum() / 3
            if bgc <= 0:
                bgc = 0.223
            elif 0 < bgc < 1:
                bgc = 1
            else:
                bgc = 0
            renderer.SetBackground(bgc, bgc, bgc)

        elif key == "8":
            bg2cols = [
                "lightyellow",
                "darkseagreen",
                "palegreen",
                "steelblue",
                "lightblue",
                "cadetblue",
                "lavender",
                "white",
                "blackboard",
                "black",
            ]
            bg2name = vedo.get_color_name(renderer.GetBackground2())
            if bg2name in bg2cols:
                idx = bg2cols.index(bg2name)
            else:
                idx = 4
            if idx is not None:
                bg2name_next = bg2cols[(idx + 1) % (len(bg2cols) - 1)]
            if not bg2name_next:
                renderer.GradientBackgroundOff()
            else:
                renderer.GradientBackgroundOn()
                renderer.SetBackground2(vedo.get_color(bg2name_next))

        elif key in ["plus", "equal", "KP_Add", "minus", "KP_Subtract"]:  # cycle axes style
            i = self.renderers.index(renderer)
            try:
                self.axes_instances[i].EnabledOff()
                self.axes_instances[i].SetInteractor(None)
            except AttributeError:
                # print("Cannot remove widget", [self.axes_instances[i]])
                try:
                    self.remove(self.axes_instances[i])
                except:
                    print("Cannot remove axes", [self.axes_instances[i]])
                    return
            self.axes_instances[i] = None

            if not self.axes:
                self.axes = 0
            if isinstance(self.axes, dict):
                self.axes = 1

            if key in ["minus", "KP_Subtract"]:
                if not self.camera.GetParallelProjection() and self.axes == 0:
                    self.axes -= 1  # jump ruler doesnt make sense in perspective mode
                bns = self.renderer.ComputeVisiblePropBounds()
                addons.add_global_axes(axtype=(self.axes - 1) % 15, c=None, bounds=bns)
            else:
                if not self.camera.GetParallelProjection() and self.axes == 12:
                    self.axes += 1  # jump ruler doesnt make sense in perspective mode
                bns = self.renderer.ComputeVisiblePropBounds()
                addons.add_global_axes(axtype=(self.axes + 1) % 15, c=None, bounds=bns)
            self.render()

        elif "KP_" in key or key in [
                "Insert","End","Down","Next","Left","Begin","Right","Home","Up","Prior"
            ]:
            asso = {  # change axes style
                "KP_Insert": 0, "KP_0": 0, "Insert": 0,
                "KP_End":    1, "KP_1": 1, "End":    1,
                "KP_Down":   2, "KP_2": 2, "Down":   2,
                "KP_Next":   3, "KP_3": 3, "Next":   3,
                "KP_Left":   4, "KP_4": 4, "Left":   4,
                "KP_Begin":  5, "KP_5": 5, "Begin":  5,
                "KP_Right":  6, "KP_6": 6, "Right":  6,
                "KP_Home":   7, "KP_7": 7, "Home":   7,
                "KP_Up":     8, "KP_8": 8, "Up":     8,
                "Prior":     9,  # on windows OS
            }
            clickedr = self.renderers.index(renderer)
            if key in asso:
                if self.axes_instances[clickedr]:
                    if hasattr(self.axes_instances[clickedr], "EnabledOff"):  # widget
                        self.axes_instances[clickedr].EnabledOff()
                    else:
                        try:
                            renderer.RemoveActor(self.axes_instances[clickedr])
                        except:
                            pass
                    self.axes_instances[clickedr] = None
                bounds = renderer.ComputeVisiblePropBounds()
                addons.add_global_axes(axtype=asso[key], c=None, bounds=bounds)
                self.interactor.Render()

        if key == "O":
            renderer.RemoveLight(self._extralight)
            self._extralight = None

        elif key == "o":
            vbb, sizes, _, _ = addons.compute_visible_bounds()
            cm = utils.vector((vbb[0] + vbb[1]) / 2, (vbb[2] + vbb[3]) / 2, (vbb[4] + vbb[5]) / 2)
            if not self._extralight:
                vup = renderer.GetActiveCamera().GetViewUp()
                pos = cm + utils.vector(vup) * utils.mag(sizes)
                self._extralight = addons.Light(pos, focal_point=cm, intensity=0.4)
                renderer.AddLight(self._extralight)
                vedo.printc("Press 'o' again to rotate light source, or 'O' to remove it.", c='y')
            else:
                cpos = utils.vector(self._extralight.GetPosition())
                x, y, z = self._extralight.GetPosition() - cm
                r, th, ph = transformations.cart2spher(x, y, z)
                th += 0.2
                if th > np.pi:
                    th = np.random.random() * np.pi / 2
                ph += 0.3
                cpos = transformations.spher2cart(r, th, ph).T + cm
                self._extralight.SetPosition(cpos)

        elif key == "l":
            if self.clicked_object in self.get_meshes():
                objs = [self.clicked_object]
            else:
                objs = self.get_meshes()
            for ia in objs:
                try:
                    ev = ia.properties.GetEdgeVisibility()
                    ia.properties.SetEdgeVisibility(not ev)
                    ia.properties.SetRepresentationToSurface()
                    ia.properties.SetLineWidth(0.1)
                except AttributeError:
                    pass

        elif key == "k":  # lightings
            if self.clicked_object in self.get_meshes():
                objs = [self.clicked_object]
            else:
                objs = self.get_meshes()
            shds = ("default", "metallic", "plastic", "shiny", "glossy", "off")
            for ia in objs:
                try:
                    lnr = (ia._ligthingnr + 1) % 6
                    ia.lighting(shds[lnr])
                    ia._ligthingnr = lnr
                except AttributeError:
                    pass

        elif key == "K":  # shading
            if self.clicked_object in self.get_meshes():
                objs = [self.clicked_object]
            else:
                objs = self.get_meshes()
            for ia in objs:
                if isinstance(ia, vedo.Mesh):
                    ia.compute_normals(cells=False)
                    intrp = ia.properties.GetInterpolation()
                    if intrp > 0:
                        ia.properties.SetInterpolation(0)  # flat
                    else:
                        ia.properties.SetInterpolation(2)  # phong

        elif key == "n":  # show normals to an actor
            self.remove("added_auto_normals")
            if self.clicked_object in self.get_meshes():
                if self.clicked_actor.GetPickable():
                    norml = vedo.shapes.NormalLines(self.clicked_object)
                    norml.name = "added_auto_normals"
                    self.add(norml)

        elif key == "x":
            if self.justremoved is None:
                if self.clicked_object in self.get_meshes() or isinstance(
                        self.clicked_object, vtki.vtkAssembly
                    ):
                    self.justremoved = self.clicked_actor
                    self.renderer.RemoveActor(self.clicked_actor)
            else:
                self.renderer.AddActor(self.justremoved)
                self.justremoved = None

        elif key == "X":
            if self.clicked_object:
                if not self.cutter_widget:
                    self.cutter_widget = addons.BoxCutter(self.clicked_object)
                    self.add(self.cutter_widget)
                    vedo.printc("Press i to toggle the cutter on/off", c='g', dim=1)
                    vedo.printc("      u to flip selection", c='g', dim=1)
                    vedo.printc("      r to reset cutting planes", c='g', dim=1)
                    vedo.printc("      Shift+X to close the cutter box widget", c='g', dim=1)
                    vedo.printc("      Ctrl+S to save the cut section to file.", c='g', dim=1)
                else:
                    self.remove(self.cutter_widget)
                    self.cutter_widget = None
                vedo.printc("Click object and press X to open the cutter box widget.", c='g')

        elif key == "E":
            vedo.printc(r":camera: Exporting 3D window to file scene.npz", c="b", end="")
            vedo.file_io.export_window("scene.npz")
            vedo.printc(", try:\n> vedo scene.npz  # (this is experimental!)", c="b")
            return

        elif key == "F":
            vedo.file_io.export_window("scene.x3d")
            vedo.printc(r":camera: Exporting 3D window to file", c="b", end="")
            vedo.file_io.export_window("scene.npz")
            vedo.printc(". Try:\n> firefox scene.html", c="b")

        # elif key == "G":  # not working with last version of k3d
        #     vedo.file_io.export_window("scene.html")
        #     vedo.printc(r":camera: Exporting K3D window to file", c="b", end="")
        #     vedo.file_io.export_window("scene.html")
        #     vedo.printc(". Try:\n> firefox scene.html", c="b")

        elif key == "i":  # print info
            if self.clicked_object:
                print(self.clicked_object)
            else:
                print(self)

        elif key == "I":  # print color under the mouse
            x, y = iren.GetEventPosition()
            self.color_picker([x, y], verbose=True)

        elif key == "Y":
            if self.clicked_object and self.clicked_object.pipeline:
                self.clicked_object.pipeline.show()

        if iren:
            iren.Render()
