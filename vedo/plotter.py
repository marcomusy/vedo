#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import sys
import vtk
import os.path
import numpy as np

import vedo
import vedo.utils as utils
import vedo.settings as settings
import vedo.addons as addons
import vedo.backends as backends

__doc__ = (
    """
Defines main class ``Plotter`` to manage actors and 3D rendering.
"""
    + vedo.docs._defs
)

__all__ = ["show",
           "clear",
           "Plotter",
           "interactive",
]

########################################################################
def show(*actors,
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
        azimuth=0,
        elevation=0,
        roll=0,
        camera=None,
        interactorStyle=0,
        mode=None,
        q=False,
        new=False,
    ):
    """
    Create on the fly an instance of class ``Plotter`` and show the object(s) provided.

    Allowed input objects types are:
    ``str``, ``Mesh``, ``Volume``, ``Picture``, ``Assembly``
    ``vtkPolyData``, ``vtkActor``, ``vtkActor2D``, ``vtkImageActor``,
    ``vtkAssembly`` or ``vtkVolume``.

    If filename is given, its type is guessed based on its extension.
    Supported formats are:
    `vtu, vts, vtp, ply, obj, stl, 3ds, xml, neutral, gmsh, pcd, xyz, txt, byu,
    tif, slc, vti, mhd, png, jpg`.

    :param int at: number of the renderer to plot to, if more than one exists
    :param list shape: Number of sub-render windows inside of the main window.
        Specify two across with ``shape=(2, 1)`` and a two by two grid
        with ``shape=(2, 2)``.  By default there is only one renderer.
        Can also accept a shape as string descriptor. E.g.:

          - shape="3|1" means 3 plots on the left and 1 on the right,
          - shape="4/2" means 4 plots on top of 2 at bottom.

    :param int axes: set the type of axes to be shown

            - 0,  no axes
            - 1,  draw three gray grid walls
            - 2,  show cartesian axes from (0,0,0)
            - 3,  show positive range of cartesian axes from (0,0,0)
            - 4,  show a triad at bottom left
            - 5,  show a cube at bottom left
            - 6,  mark the corners of the bounding box
            - 7,  draw a 3D ruler at each side of the cartesian axes
            - 8,  show the ``vtkCubeAxesActor`` object
            - 9,  show the bounding box outLine
            - 10, show three circles representing the maximum bounding box
            - 11, show a large grid on the x-y plane
            - 12, show polar axes
            - 13, draw a simple ruler at the bottom of the window

        Axis type-1 can be fully customized by passing a dictionary ``axes=dict()`` where:
        Check ``addons.Axes()`` for the full list of options.

    :param float azimuth/elevation/roll:  move camera accordingly
    :param str viewup:  either ['x', 'y', 'z'] or a vector to set vertical direction
    :param bool resetcam:  re-adjust camera position to fit objects
    :param dict camera: Camera parameters can further be specified with a dictionary
        assigned to the ``camera`` keyword (E.g. `show(camera={'pos':(1,2,3), 'thickness':1000,})`)

        - pos, `(list)`,  the position of the camera in world coordinates

        - focalPoint `(list)`, the focal point of the camera in world coordinates

        - viewup `(list)`, the view up direction for the camera

        - distance `(float)`, set the focal point to the specified distance from the camera position.

        - clippingRange `(float)`, distance of the near and far clipping planes along the direction
            of projection.

        - parallelScale `(float)`,
            scaling used for a parallel projection, i.e. the height of the viewport
            in world-coordinate distances. The default is 1. Note that the "scale" parameter works as
            an "inverse scale", larger numbers produce smaller images.
            This method has no effect in perspective projection mode.

        - thickness `(float)`,
            set the distance between clipping planes. This method adjusts the far clipping
            plane to be set a distance 'thickness' beyond the near clipping plane.

        - viewAngle `(float)`,
            the camera view angle, which is the angular height of the camera view
            measured in degrees. The default angle is 30 degrees.
            This method has no effect in parallel projection mode.
            The formula for setting the angle up for perfect perspective viewing is:
            angle = 2*atan((h/2)/d) where h is the height of the RenderWindow
            (measured by holding a ruler up to your screen) and d is the distance
            from your eyes to the screen.

    :param bool interactive:  pause and interact with window (True)
        or continue execution (False)

    :param float rate:  maximum rate of `show()` in Hertz
    :param int mode: set the type of interaction

        - 0 = TrackballCamera [default]
        - 1 = TrackballActor
        - 2 = JoystickCamera
        - 3 = JoystickActor
        - 4 = Flight
        - 5 = RubberBand2D
        - 6 = RubberBand3D
        - 7 = RubberBandZoom
        - 8 = Context
        - 9 = 3D
        - 10 = Terrain
        - 11 = Unicam

    :param bool q:  force program to quit after `show()` command returns.

    :param bool new: if set to `True`, a call to ``show`` will instantiate
        a new ``Plotter`` object (a new window) instead of reusing the first created.

    :return: the current ``Plotter`` class instance.

    .. note:: With multiple renderers, keyword ``at`` can become a `list`, e.g.

        .. code-block:: python

            from vedo import *
            s = Sphere()
            c = Cube()
            p = Paraboloid()
            show(s, c, at=[0, 1], shape=(3,1))
            show(p, at=2, interactive=True)
            #
            # is equivalent to:
            plt = Plotter(shape=(3,1))
            s = Sphere()
            c = Cube()
            p = Paraboloid()
            plt.show(s, at=0)
            plt.show(p, at=1)
            plt.show(c, at=2, interactive=True)
    """
    if len(actors) == 0:
        actors = None
    elif len(actors) == 1:
        actors = actors[0]
    else:
        actors = utils.flatten(actors)

    if settings.plotter_instance and not new: # Plotter exists
        plt = settings.plotter_instance

    else:                                            # Plotter must be created

        if utils.isSequence(at):                     # user passed a sequence for "at"
            if not utils.isSequence(actors):
                vedo.printc("show() Error: input must be a list.", c='r')
                raise RuntimeError()
            if len(at) != len(actors):
                vedo.printc("show() Error: lists 'input' and 'at', must have equal lengths.", c='r')
                raise RuntimeError()
            if len(at) > 1 and (shape == (1, 1) and N is None):
                N = max(at) + 1

        elif at is None and (N or shape != (1, 1)):
            if not utils.isSequence(actors):
                vedo.printc('show() Error: N or shape is set, but input is not a sequence.', c='r')
                vedo.printc('              you may need to specify e.g. at=0', c='r')
                raise RuntimeError()
            at = list(range(len(actors)))

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

    # use _plt_to_return because plt.show() can return a k3d/panel plot
    _plt_to_return = None

    if utils.isSequence(at):
        for i, a in enumerate(actors):
            _plt_to_return = plt.show(  a,
                                        at=i,
                                        zoom=zoom,
                                        resetcam=resetcam,
                                        viewup=viewup,
                                        azimuth=azimuth,
                                        elevation=elevation,
                                        roll=roll,
                                        camera=camera,
                                        interactive=False,
                                        interactorStyle=interactorStyle,
                                        mode=mode,
                                        bg=bg,
                                        bg2=bg2,
                                        axes=axes,
                                        q=q,
            )
        plt.interactive = interactive

        if interactive or len(at)==N \
            or (isinstance(shape[0],int) and len(at)==shape[0]*shape[1]):
            # note that shape can be a string
            # print(interactive)
            if not offscreen and (interactive is None or interactive):
                plt.interactor.Start()

    else:

        _plt_to_return = plt.show(
                                    actors,
                                    at=at,
                                    zoom=zoom,
                                    resetcam=resetcam,
                                    viewup=viewup,
                                    azimuth=azimuth,
                                    elevation=elevation,
                                    roll=roll,
                                    camera=camera,
                                    interactive=interactive,
                                    interactorStyle=interactorStyle,
                                    mode=mode,
                                    bg=bg,
                                    bg2=bg2,
                                    axes=axes,
                                    q=q,
        )
    return _plt_to_return


def interactive():
    """Start the rendering window interaction mode."""
    if settings.plotter_instance:
        if settings.plotter_instance.escaped: # just return
            return settings.plotter_instance
        if hasattr(settings.plotter_instance, 'interactor'):
            if settings.plotter_instance.interactor:
                settings.plotter_instance.interactor.Start()
    return settings.plotter_instance


def clear(actor=None, at=None):
    """
    Clear specific actor or list of actors from the current rendering window.
    Keyword ``at`` specify the reneder to be cleared.
    """
    if not settings.plotter_instance:
        return
    settings.plotter_instance.clear(actor, at)
    return settings.plotter_instance


########################################################################
class Plotter:
    """
    Main class to manage actors.

    :param list shape: shape of the grid of renderers in format (rows, columns).
        Ignored if N is specified.
    :param int N: number of desired renderers arranged in a grid automatically.
    :param list pos: (x,y) position in pixels of top-left corner of the rendering window
        on the screen
    :param size: size of the rendering window. If 'auto', guess it based on screensize.
    :param screensize: physical size of the monitor screen
    :param bg: background color or specify jpg image file name with path
    :param bg2: background color of a gradient towards the top
    :param int axes:

      - 0,  no axes
      - 1,  draw three gray grid walls
      - 2,  show cartesian axes from (0,0,0)
      - 3,  show positive range of cartesian axes from (0,0,0)
      - 4,  show a triad at bottom left
      - 5,  show a cube at bottom left
      - 6,  mark the corners of the bounding box
      - 7,  draw a 3D ruler at each side of the cartesian axes
      - 8,  show the VTK ``CubeAxesActor`` object
      - 9,  show the bounding box outLine,
      - 10, show three circles representing the maximum bounding box,
      - 11, show a large grid on the x-y plane (use with zoom=8)
      - 12, show polar axes.
      - 13, draw a simple ruler at the bottom of the window

    Axis type-1 can be fully customized by passing a dictionary ``axes=dict()``.
    Check ``Axes()`` for the available options.

    :param bool sharecam: if False each renderer will have an independent vtkCamera
    :param bool interactive: if True will stop after show() to allow interaction w/ window
    :param bool offscreen: if True will not show the rendering window
    :param QVTKRenderWindowInteractor qtWidget:

      render in a Qt-Widget using an QVTKRenderWindowInteractor.
      Overrides offscreen to True
      Overrides interactive to False
      See Also: examples qt_windows1.py and qt_windows2.py

    |multiwindows|
    """

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
        qtWidget=None,
        wxWidget=None,
        ):

        settings.plotter_instance = self
        settings.plotter_instances.append(self)

        if qtWidget is not None:
            # overrides the interactive and offscreen properties
            interactive = False
            offscreen = True

        if wxWidget is not None:
            # overrides the interactive property
            interactive = False

        if interactive is None:
            if N==1:
                interactive = True
            elif N or shape != (1, 1):
                interactive = False
            else:
                interactive = True

        self.actors = []  # list of actors to be shown
        self.clickedActor = None  # holds the actor that has been clicked
        self.renderer = None  # current renderer
        self.renderers = []  # list of renderers
        self.shape = shape  # don't remove this line
        self.interactive = interactive  # allows to interact with renderer
        self.axes = axes  # show axes type nr.
        self.title = title  # window title
        self.sharecam = sharecam  # share the same camera if multiple renderers
        self.picker = None    # the vtkPicker object
        self.picked2d = None  # 2d coords of a clicked point on the rendering window
        self.picked3d = None  # 3d coords of a clicked point on an actor
        self.offscreen = offscreen
        self.resetcam = resetcam

        self.qtWidget = qtWidget #  QVTKRenderWindowInteractor
        self.wxWidget = wxWidget # wxVTKRenderWindowInteractor

        self.skybox = None
        self.frames = None      # holds the output of addons.addRendererFrame

        # mostly internal stuff:
        self.hoverLegends = []
        self.backgrcol = bg
        self.pos = pos     # used by vedo.io
        self.justremoved = None
        self.axes_instances = []
        self._icol = 0
        self.clock = 0
        self._clockt0 = time.time()
        self.sliders = []
        self.buttons = []
        self.widgets = []
        self.cutterWidget = None
        self.flagWidget = None
        self._flagRep = None
        self.scalarbars = []
        self.backgroundRenderer = None
        self.keyPressFunction = None         # obsolete! use plotter.callBack()
        self.mouseLeftClickFunction = None   # obsolete! use plotter.callBack()
        self.mouseMiddleClickFunction = None # obsolete! use plotter.callBack()
        self.mouseRightClickFunction = None  # obsolete! use plotter.callBack()
        self._first_viewup = True
        self._extralight = None
        self.size = size
        self.interactor = None
        self.keyheld = ''
        self.xtitle = settings.xtitle  # x axis label and units
        self.ytitle = settings.ytitle  # y axis label and units
        self.ztitle = settings.ztitle  # z axis label and units

        # build the rendering window:
        self.camera = vtk.vtkCamera()
        self.window = vtk.vtkRenderWindow()
        self.escaped = False

        self.window.GlobalWarningDisplayOff()

        self._repeating_timer_id = None
        self._timer_event_id = None

        ############################################################
        notebookBackend = settings.notebookBackend
        if notebookBackend:
            if notebookBackend == '2d':
                self.offscreen = True
                if self.size == "auto":
                    self.size = (900, 700)

            elif notebookBackend == "k3d" or "ipygany" in notebookBackend:
                self.interactive = False
                self.interactor = None
                self.window = None
                self.camera = None # let the backend choose
                if self.size == "auto":
                    self.size = (1000, 1000)
                ############################
                return #####################
                ############################

        # more settings
        if settings.useDepthPeeling:
            self.window.SetAlphaBitPlanes(settings.alphaBitPlanes)
        self.window.SetMultiSamples(settings.multiSamples)

        self.window.SetPolygonSmoothing(settings.polygonSmoothing)
        self.window.SetLineSmoothing(settings.lineSmoothing)
        self.window.SetPointSmoothing(settings.pointSmoothing)

        # sort out screen size
        if screensize == "auto":
            screensize = (2160, 1440) # might go wrong, use a default 1.5 ratio
            ### BUG in GetScreenSize https://discourse.vtk.org/t/vtk9-1-0-problems/7094/3
            # vtkvers = settings.vtk_version
            # if not self.offscreen and (vtkvers[0]<9 or vtkvers[0]==9 and vtkvers[1]==0):
            # if False:
            #     aus = self.window.GetScreenSize()
            #     if aus and len(aus) == 2 and aus[0] > 100 and aus[1] > 100:  # seems ok
            #         if aus[0] / aus[1] > 2:  # looks like there are 2 or more screens
            #             screensize = (int(aus[0] / 2), aus[1])
            #         else:
            #             screensize = aus
        x, y = screensize

        if N:  # N = number of renderers. Find out the best
            if shape != (1, 1):  # arrangement based on minimum nr. of empty renderers
                vedo.printc("Warning: having set N, shape is ignored.", c='r')
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

            if '|' in shape:
                if self.size == "auto":
                    self.size = (800, 1200)
                n = int(shape.split('|')[0])
                m = int(shape.split('|')[1])
                rangen = reversed(range(n))
                rangem = reversed(range(m))
            else:
                if self.size == "auto":
                    self.size = (1200, 800)
                m = int(shape.split('/')[0])
                n = int(shape.split('/')[1])
                rangen = range(n)
                rangem = range(m)

            if n>=m:
                xsplit = m/(n+m)
            else:
                xsplit = 1-n/(n+m)
            if settings.windowSplittingPosition:
                xsplit = settings.windowSplittingPosition

            for i in rangen:
                arenderer = vtk.vtkRenderer()
                if '|' in shape:
                    arenderer.SetViewport(0,  i/n, xsplit, (i+1)/n)
                else:
                    arenderer.SetViewport(i/n, 0,  (i+1)/n, xsplit )
                self.renderers.append(arenderer)

            for i in rangem:
                arenderer = vtk.vtkRenderer()

                if '|' in shape:
                    arenderer.SetViewport(xsplit, i/m, 1, (i+1)/m)
                else:
                    arenderer.SetViewport(i/m, xsplit, (i+1)/m, 1)
                self.renderers.append(arenderer)

            for r in self.renderers:
                r.SetUseHiddenLineRemoval(settings.hiddenLineRemoval)
                r.SetLightFollowCamera(settings.lightFollowsCamera)

                r.SetUseDepthPeeling(settings.useDepthPeeling)
                #r.SetUseDepthPeelingForVolumes(settings.useDepthPeeling)
                if settings.useDepthPeeling:
                    r.SetMaximumNumberOfPeels(settings.maxNumberOfPeels)
                    r.SetOcclusionRatio(settings.occlusionRatio)
                r.SetUseFXAA(settings.useFXAA)
                r.SetPreserveDepthBuffer(settings.preserveDepthBuffer)
#                if hasattr(r, "SetUseSSAO"):
#                    r.SetUseSSAO(settings.useSSAO)
#                    r.SetSSAORadius(settings.SSAORadius)
#                    r.SetSSAOBias(settings.SSAOBias)
#                    r.SetSSAOKernelSize(settings.SSAOKernelSize)
#                    r.SetSSAOBlur(settings.SSAOBlur)


                r.SetBackground(vedo.getColor(self.backgrcol))

                self.axes_instances.append(None)

            self.shape = (n+m,)

        elif utils.isSequence(shape) and isinstance(shape[0], dict):
            # passing a sequence of dicts for renderers specifications

            if self.size == "auto":
                self.size = (1200,900)

            for rd in shape:
                x0, y0 = rd['bottomleft']
                x1, y1 = rd['topright']
                bg_ = rd.pop('bg', 'white')
                bg2_ = rd.pop('bg2', None)

                arenderer = vtk.vtkRenderer()
                arenderer.SetUseHiddenLineRemoval(settings.hiddenLineRemoval)
                arenderer.SetLightFollowCamera(settings.lightFollowsCamera)

                arenderer.SetUseDepthPeeling(settings.useDepthPeeling)
                #arenderer.SetUseDepthPeelingForVolumes(settings.useDepthPeeling)
                if settings.useDepthPeeling:
                    arenderer.SetMaximumNumberOfPeels(settings.maxNumberOfPeels)
                    arenderer.SetOcclusionRatio(settings.occlusionRatio)
                arenderer.SetUseFXAA(settings.useFXAA)
                arenderer.SetPreserveDepthBuffer(settings.preserveDepthBuffer)
#                if hasattr(arenderer, "SetUseSSAO"):
#                    arenderer.SetUseSSAO(settings.useSSAO)
#                    arenderer.SetSSAORadius(settings.SSAORadius)
#                    arenderer.SetSSAOBias(settings.SSAOBias)
#                    arenderer.SetSSAOKernelSize(settings.SSAOKernelSize)
#                    arenderer.SetSSAOBlur(settings.SSAOBlur)

                arenderer.SetViewport(x0, y0, x1, y1)
                arenderer.SetBackground(vedo.getColor(bg_))
                if bg2_:
                    arenderer.GradientBackgroundOn()
                    arenderer.SetBackground2(vedo.getColor(bg2_))

                self.renderers.append(arenderer)
                self.axes_instances.append(None)

            self.shape = (len(shape),)

        else:

            if self.size == "auto":  # figure out a reasonable window size
                f = 1.5
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

            image_actor=None
            bgname = str(self.backgrcol).lower()
            if ".jpg" in bgname or ".jpeg" in bgname or ".png" in bgname:
                self.window.SetNumberOfLayers(2)
                self.backgroundRenderer = vtk.vtkRenderer()
                self.backgroundRenderer.SetLayer(0)
                self.backgroundRenderer.InteractiveOff()
                self.backgroundRenderer.SetBackground(vedo.getColor(bg2))
                image_actor = vedo.Picture(self.backgrcol)
                self.window.AddRenderer(self.backgroundRenderer)
                self.backgroundRenderer.AddActor(image_actor)

            for i in reversed(range(shape[0])):
                for j in range(shape[1]):
                    arenderer = vtk.vtkRenderer()
                    arenderer.SetUseHiddenLineRemoval(settings.hiddenLineRemoval)
                    arenderer.SetLightFollowCamera(settings.lightFollowsCamera)
                    arenderer.SetTwoSidedLighting(settings.twoSidedLighting)

                    arenderer.SetUseDepthPeeling(settings.useDepthPeeling)
                    #arenderer.SetUseDepthPeelingForVolumes(settings.useDepthPeeling)
                    if settings.useDepthPeeling:
                        arenderer.SetMaximumNumberOfPeels(settings.maxNumberOfPeels)
                        arenderer.SetOcclusionRatio(settings.occlusionRatio)
                    arenderer.SetUseFXAA(settings.useFXAA)
                    arenderer.SetPreserveDepthBuffer(settings.preserveDepthBuffer)
#                    if hasattr(arenderer, "SetUseSSAO"):
#                        arenderer.SetUseSSAO(settings.useSSAO)
#                        arenderer.SetSSAORadius(settings.SSAORadius)
#                        arenderer.SetSSAOBias(settings.SSAOBias)
#                        arenderer.SetSSAOKernelSize(settings.SSAOKernelSize)
#                        arenderer.SetSSAOBlur(settings.SSAOBlur)

                    if image_actor:
                        arenderer.SetLayer(1)

                    arenderer.SetBackground(vedo.getColor(self.backgrcol))
                    if bg2:
                        arenderer.GradientBackgroundOn()
                        arenderer.SetBackground2(vedo.getColor(bg2))

                    x0 = i / shape[0]
                    y0 = j / shape[1]
                    x1 = (i + 1) / shape[0]
                    y1 = (j + 1) / shape[1]
                    arenderer.SetViewport(y0, x0, y1, x1)
                    self.renderers.append(arenderer)
                    self.axes_instances.append(None)
            self.shape = shape

        if len(self.renderers):
            self.renderer = self.renderers[0]

        if self.size[0] == 'f':  # full screen
            self.size = 'fullscreen'
            self.window.SetFullScreen(True)
            self.window.BordersOn()
        else:
            self.window.SetSize(int(self.size[0]), int(self.size[1]))

        if self.wxWidget is not None:
            settings.immediateRendering = False # overrride
            self.window = self.wxWidget.GetRenderWindow() # overwrite
            self.interactor = self.window.GetInteractor()
            for r in self.renderers:
                self.window.AddRenderer(r)
            self.wxWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.camera = self.renderer.GetActiveCamera()
            # if settings.enableDefaultMouseCallbacks:
            #     self.wxWidget.AddObserver("LeftButtonPressEvent", self._mouseleft)
            #     self.wxWidget.AddObserver("RightButtonPressEvent", self._mouseright)
            #     self.wxWidget.AddObserver("MiddleButtonPressEvent", self._mousemiddle)
            # if settings.enableDefaultKeyboardCallbacks:
            #     self.wxWidget.AddObserver("KeyPressEvent", self._keypress)
            #     self.wxWidget.AddObserver("KeyReleaseEvent", self._keyrelease)
            ########################
            return #################
            ########################

        if self.qtWidget is not None:
            self.interactor = self.qtWidget.GetRenderWindow().GetInteractor()
            self.window = self.qtWidget.GetRenderWindow() # overwrite
            ########################
            return #################
            ########################

        self.window.SetPosition(pos)

        for r in self.renderers:
            self.window.AddRenderer(r)

        if self.offscreen:
            if self.axes == 4 or self.axes == 5:
                self.axes = 0 #doesn't work with those
            self.window.SetOffScreenRendering(True)
            self.interactive = False
            self.interactor = None
            ########################
            return #################
            ########################

        if settings.notebookBackend == "panel":
            ########################
            return #################
            ########################

        self.interactor = vtk.vtkRenderWindowInteractor()
        # self.interactor.EnableRenderOff()

        self.interactor.SetRenderWindow(self.window)
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(vsty)

        # this is causing crash in clone_viewer if disabled :((
        if settings.enableDefaultMouseCallbacks:
            self.interactor.AddObserver("LeftButtonPressEvent", self._mouseleft)
            self.interactor.AddObserver("RightButtonPressEvent", self._mouseright)
            self.interactor.AddObserver("MiddleButtonPressEvent", self._mousemiddle)
        if settings.enableDefaultKeyboardCallbacks:
            self.interactor.AddObserver("KeyPressEvent", self._keypress)
            self.interactor.AddObserver("KeyReleaseEvent", self._keyrelease)

        if settings.allowInteraction:
            def win_interact(iren, event):  # flushing interactor events
                if event == "TimerEvent":
                    iren.ExitCallback()
            self._timer_event_id = self.interactor.AddObserver("TimerEvent", win_interact)

        return ##############################################################
        ##################################################################### ..init ends here.


    def allowInteraction(self):
        """Call this method from inside a loop to allow mouse and keyboard interaction."""
        if self.interactor and self._timer_event_id is not None and settings.immediateRendering:
            self._repeatingtimer_id = self.interactor.CreateRepeatingTimer(10)
            self.interactor.Start()
            if self.interactor:
                self.interactor.DestroyTimer(self._repeatingtimer_id)
            self._repeatingtimer_id = None
        return self

    def __iadd__(self, actors):
        self.add(actors, render=False)
        return self

    def __isub__(self, actors):
        self.remove(actors, render=False)
        return self

    def load(self, filename, unpack=True, force=False):
        """
        Load objects from file.
        The output will depend on the file extension. See examples below.

        :param bool unpack: only for multiblock data,
            if True returns a flat list of objects.
        :param bool force: when downloading a file ignore any previous
            cached downloads and force a new one.

        :Example:

            .. code-block:: python

                from vedo import *
                # Return a list of 2 Mesh
                g = load([dataurl+'250.vtk', dataurl+'290.vtk'])
                show(g)
                # Return a list of meshes by reading all files in a directory
                # (if directory contains DICOM files then a Volume is returned)
                g = load('mydicomdir/')
                show(g)
                # Return a Volume. Color/Opacity transfer function can be specified too.
                g = load(dataurl+'embryo.slc')
                g.c(['y','lb','w']).alpha((0.0, 0.4, 0.9, 1)).show()
        """
        acts = vedo.io.load(filename, unpack, force)
        if utils.isSequence(acts):
            self.actors += acts
        else:
            self.actors.append(acts)
        return acts

    def add(self, actors, at=None, render=True, resetcam=False):
        """Append input object to the internal list of actors to be shown.

        :param int at: add the object at the specified renderer
        :param bool render: render the scene after adding the object
        """
        if at is not None:
            self.renderer = self.renderers[at]

        actors = self._scan_input(actors)

        if utils.isSequence(actors):
            for a in actors:
                if a not in self.actors:
                    self.actors.append(a)
                if self.renderer:
                    self.renderer.AddActor(a)
        else:
            self.actors.append(actors)
            self.renderer.AddActor(actors)
        if render:
            self.render(resetcam=resetcam)
        return self


    def remove(self, actors, at=None, render=False, resetcam=False):
        """Remove input object to the internal list of actors to be shown.

        :param int at: remove the object at the specified renderer
        :param bool render: render the scene after removing the object
        """
        if at is not None:
            ren = self.renderers[at]
        else:
            ren = self.renderer
        if not utils.isSequence(actors):
            actors = [actors]

        for a in actors:

            if isinstance(a, str):
                for b in self.actors:
                    if hasattr(b, "name") and a in b.name:
                        a = b
                        break
                if isinstance(a, str):
                    continue # did not find that name so skip

            if ren:
                ren.RemoveActor(a)
                if hasattr(a, 'renderedAt'):
                    ir = self.renderers.index(ren)
                    a.renderedAt.discard(ir)
                if hasattr(a, 'scalarbar') and a.scalarbar:
                    ren.RemoveActor(a.scalarbar)
                if hasattr(a, 'trail') and a.trail:
                    ren.RemoveActor(a.trail)
                    a.trailPoints = []
            if a in self.actors:
                i = self.actors.index(a)
                del self.actors[i]
        if render:
            self.render(resetcam=resetcam)
        return self

    def pop(self, at=0):
        """Remove the last added object from the rendering window"""
        self.remove(self.actors[-1], at=at)
        return self

    def render(self, at=None, resetcam=False):
        """Render the scene."""
        if not self.window:
            return self

        if self.wxWidget:
            if resetcam:
                self.renderer.ResetCamera()
            self.wxWidget.Render()
            return self

        if self.qtWidget:
            if resetcam:
                self.renderer.ResetCamera()
            self.qtWidget.Render()
            return self

        if self.interactor:
            if not self.interactor.GetInitialized():
                self.interactor.Initialize()

        if at is not None: # disable all except i==at
            self.window.EraseOff()
            if at < 0:
                at = at + len(self.renderers) +1
            for i in range(len(self.renderers)):
                if i != at:
                    self.renderers[i].DrawOff()

        if settings.vtk_version[0] == 9 and "Darwin" in settings.sys_platform:
            for a in self.actors:
                if isinstance(a, vtk.vtkVolume):
                    self.window.SetMultiSamples(0) # to fix mac OSX BUG vtk9
                    break
        if resetcam:
            self.renderer.ResetCamera()

        self.window.Render()

        if at is not None: # re-enable all that were disabled
            for i in range(len(self.renderers)):
                if i != at:
                    self.renderers[i].DrawOn()
            self.window.EraseOn()

        return self

    def enableErase(self, value=True):
        """Enable erasing the redering window between render() calls."""
        self.window.SetErase(value)
        return self

    def enableRenderer(self, at, value=True):
        """Enable a render() call to refresh this renderer."""
        self.renderers[at].SetDraw(value)
        return self

    def useDepthPeeling(self, at, value=True):
        """
        Specify whether use depth peeling algorithm at this specific renderer
        Call this method before the first rendering.
        """
        self.renderers[at].SetUseDepthPeeling(value)
        return self

    def background(self, c1=None, c2=None, at=None):
        """Set the color of the background for the current renderer.
        A different renderer index can be specified by keyword ``at``.

        Parameters
        ----------
        c1 : list, optional
            background main color. The default is None.
        c2 : list, optional
            background color for the upper part of the window.
            The default is None.
        at : int, optional
            renderer index. The default is 0.
        """
        if not len(self.renderers):
            return self
        if at is None:
            r = self.renderer
        else:
            r = self.renderers[at]
        if r:
            if c1 is not None:
                r.SetBackground(vedo.getColor(c1))
            if c2 is not None:
                r.GradientBackgroundOn()
                r.SetBackground2(vedo.getColor(c2))
            else:
                r.GradientBackgroundOff()
        return self

    # def addShadows(self, at=0):
    #     """to do"""
    #     smp = vtk.vtkShadowMapPass()
    #     rpc = vtk.vtkRenderPassCollection()
    #     rpc.AddItem(smp.GetShadowMapBakerPass())
    #     rpc.AddItem(smp)

    #     seq = vtk.vtkSequencePass()
    #     seq.SetPasses(rpc)
    #     cpass = vtk.vtkCameraPass()
    #     cpass.SetDelegatePass(seq)
    #     self.renderers[at].SetPass(cpass)
    #     self.renderers[at].Modified()
    #     return self


    ####################################################
    def getMeshes(self, at=None, includeNonPickables=False):
        """
        Return a list of Meshes from the specified renderer.

        :param int at: specify which renderer to look into.
        :param bool includeNonPickables: include non-pickable objects
        """
        if at is None:
            renderer = self.renderer
            at=0
        elif isinstance(at, int):
            renderer = self.renderers[at]

        has_global_axes = False
        if isinstance(self.axes_instances[at], vedo.Assembly):
            has_global_axes=True

        actors = []
        acs = renderer.GetActors()
        acs.InitTraversal()
        for i in range(acs.GetNumberOfItems()):
            a = acs.GetNextItem()
            if isinstance(a, vtk.vtkVolume):
                continue
            if includeNonPickables or a.GetPickable():
                if a == self.axes_instances[at]:
                    continue
                if has_global_axes and a in self.axes_instances[at].actors:
                    continue
                actors.append(a)
        return actors

    def getVolumes(self, at=None, includeNonPickables=False):
        """
        Return a list of Volumes from the specified renderer.

        :param int at: specify which renderer to look into.
        :param bool includeNonPickables: include non-pickable objects
        """
        if at is None:
            renderer = self.renderer
            at=0
        elif isinstance(at, int):
            renderer = self.renderers[at]

        vols = []
        acs = renderer.GetVolumes()
        acs.InitTraversal()
        for i in range(acs.GetNumberOfItems()):
            a = acs.GetNextItem()
            if includeNonPickables or a.GetPickable():
                vols.append(a)
        return vols


    def resetCamera(self):
        """Reset the camera position and zooming."""
        self.renderer.ResetCamera()
        return self

    def moveCamera(self, camstart, camstop, fraction):
        """
        Takes as input two ``vtkCamera`` objects and set camera at an intermediate position:

        fraction=0 -> camstart,  fraction=1 -> camstop.

        ``camstart`` and ``camstop`` can also be dictionaries of format:

            camstart = dict(pos=..., focalPoint=..., viewup=..., distance=..., clippingRange=...)

        Press ``shift-C`` key in interactive mode to dump a python snipplet
        of parameters for the current camera view.
        """
        if fraction > 1:
            fraction = 1
        if fraction < 0:
            fraction = 0

        if isinstance(camstart, dict):
            p1 = np.asarray(camstart.pop("pos", [0,0,1]))
            f1 = np.asarray(camstart.pop("focalPoint", [0,0,0]))
            v1 = np.asarray(camstart.pop("viewup", [0,1,0]))
            s1 = camstart.pop("distance", None)
            c1 = np.asarray(camstart.pop("clippingRange", None))
        else:
            p1 = np.array(camstart.GetPosition())
            f1 = np.array(camstart.GetFocalPoint())
            v1 = np.array(camstart.GetViewUp())
            c1 = np.array(camstart.GetClippingRange())
            s1 = camstart.GetDistance()

        if isinstance(camstop, dict):
            p2 = np.asarray(camstop.pop("pos", [0,0,1]))
            f2 = np.asarray(camstop.pop("focalPoint", [0,0,0]))
            v2 = np.asarray(camstop.pop("viewup", [0,1,0]))
            s2 = camstop.pop("distance", None)
            c2 = np.asarray(camstop.pop("clippingRange", None))
        else:
            p2 = np.array(camstop.GetPosition())
            f2 = np.array(camstop.GetFocalPoint())
            v2 = np.array(camstop.GetViewUp())
            c2 = np.array(camstop.GetClippingRange())
            s2 = camstop.GetDistance()

        ufraction = 1 - fraction
        self.camera.SetPosition(  p2 * fraction + p1 * ufraction)
        self.camera.SetFocalPoint(f2 * fraction + f1 * ufraction)
        self.camera.SetViewUp    (v2 * fraction + v1 * ufraction)
        if s1 is not None and s2 is not None :
            self.camera.SetDistance(s2 * fraction + s1 * ufraction)
        if c1 is not None and c2 is not None:
            self.camera.SetClippingRange(c2 * fraction + c1 * ufraction)
        return self

    def flyTo(self, point, at=0):
        """
        Fly camera to the specified point.

        Parameters
        ----------
        point : list
            point in space to place camera.
        at : int, optional
            Renderer number. The default is 0.

        Example:

            .. code-block:: python

                from vedo import Cone
                Cone().show(axes=1).flyTo([1,0,0]).show()
        """
        self.resetcam = False
        self.interactor.FlyTo(self.renderers[at], point)
        return self


    def record(self, filename='.vedo_recorded_events.log'):
        """
        Record camera, mouse, keystrokes and all other events.
        Recording can be toggled on/off by pressing key "R".

        Parameters
        ----------
        filename : str, optional
            ascii file to store events. The default is '.vedo_recorded_events.log'.

        Returns
        -------
        events : str
            a string descriptor of events.
        """
        erec = vtk.vtkInteractorEventRecorder()
        erec.SetInteractor(self.interactor)
        erec.SetFileName(filename)
        erec.SetKeyPressActivationValue("R")
        erec.EnabledOn()
        erec.Record()
        self.interactor.Start()
        erec.Stop()
        erec.EnabledOff()
        with open(filename, 'r') as fl:
            events = fl.read()
        erec = None
        return events

    def play(self, events='.vedo_recorded_events.log', repeats=0):
        """
        Play camera, mouse, keystrokes and all other events.

        Parameters
        ----------
        events : str, optional
            file o string of events. The default is '.vedo_recorded_events.log'.
        repeats : int, optional
            number of extra repeats of the same events. The default is 0.
        """
        erec = vtk.vtkInteractorEventRecorder()
        erec.SetInteractor(self.interactor)

        if events.endswith(".log"):
            erec.ReadFromInputStringOff()
            erec.SetFileName(events)
        else:
            erec.ReadFromInputStringOn()
            erec.SetInputString(events)

        erec.Play()
        for _i in range(repeats):
            erec.Rewind()
            erec.Play()
        erec.EnabledOff()
        erec = None
        return self


    def parallelProjection(self, value=True, at=0):
        """
        Use parallel projection ``at`` a specified renderer.
        Object is seen from "infinite" distance, e.i. remove any perspective effects.
        """
        r = self.renderers[at]
        r.GetActiveCamera().SetParallelProjection(value)
        r.Modified()
        return self


    ##################################################################
    def addSlider2D(self, sliderfunc, xmin, xmax,
                    value=None, pos=4, title="", font="", titleSize=1, c=None,
                    showValue=True, delayed=False):
        """Add a slider widget which can call an external custom function.

        :param sliderfunc: external function to be called by the widget
        :param float xmin:  lower value
        :param float xmax:  upper value
        :param float value: current value
        :param list pos: position corner number: horizontal [1-5] or vertical [11-15]
            it can also be specified by corners coordinates [(x1,y1), (x2,y2)]
        :param str title: title text
        :param float titleSize: title text scale [1.0]
        :param str font: title font [arial, courier]
        :param bool showValue: if true current value is shown
        :param bool delayed: if True the callback is delayed to when the mouse is released

        |sliders1| |sliders1.py|_ |sliders2.py|_
        """
        return addons.addSlider2D(sliderfunc, xmin, xmax, value,
                                  pos, title, font, titleSize, c, showValue, delayed)

    def addSlider3D(
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
        rotation=0,
        c=None,
        showValue=True,
    ):
        """Add a 3D slider widget which can call an external custom function.

        :param sliderfunc: external function to be called by the widget
        :param list pos1: first position coordinates
        :param list pos2: second position coordinates
        :param float xmin:  lower value
        :param float xmax:  upper value
        :param float value: initial value
        :param float s: label scaling factor
        :param float t: tube scaling factor
        :param str title: title text
        :param c: slider color
        :param float rotation: title rotation around slider axis
        :param bool showValue: if True current value is shown

        |sliders3d| |sliders3d.py|_
        """
        return addons.addSlider3D(
            sliderfunc, pos1, pos2, xmin, xmax, value, s, t, title, rotation, c, showValue
        )

    def addButton(
        self,
        fnc,
        states=("On", "Off"),
        c=("w", "w"),
        bc=("dg", "dr"),
        pos=(0.7, 0.05),
        size=24,
        font="Normografo",
        bold=False,
        italic=False,
        alpha=1,
        angle=0,
    ):
        """Add a button to the renderer window.

        :param list states: a list of possible states, e.g. ['On', 'Off']
        :param c: a list of colors for each state
        :param bc: a list of background colors for each state
        :param pos: 2D position in pixels from left-bottom corner
        :param size: size of button font
        :param str font: font type (arial, courier, times)
        :param bool bold: bold face (False)
        :param bool italic: italic face (False)
        :param float alpha: opacity level
        :param float angle: anticlockwise rotation in degrees

        |buttons| |buttons.py|_
        """
        return addons.addButton(fnc, states, c, bc, pos, size, font,
                                bold, italic, alpha, angle)

    def addSplineTool(self, points, pc='k', ps=8, lc='r4', ac='g5', lw=2, closed=False, interactive=True):
        """
        Add a spline tool to the current plotter. Nodes of the spline can be dragged in space
        with the mouse.
        Clicking on the line itself adds an extra point.
        Selecting a point and pressing ``del`` removes it.

        Parameters
        ----------
        points : Mesh, Points, array
            the set of vertices forming the spline nodes.
        pc : str, optional
            point color. The default is 'k'.
        ps : str, optional
            point size. The default is 8.
        lc : str, optional
            line color. The default is 'r4'.
        ac : str, optional
            active point marker color. The default is 'g5'.
        lw : int, optional
            line width. The default is 2.
        closed : bool, optional
            spline is meant to be closed. The default is False.

        Returns
        -------
        SplineTool object.
        """
        sw = addons.SplineTool(points, pc, ps, lc, ac, lw, closed)
        if self.interactor:
            sw.SetInteractor(self.interactor)
        else:
            vedo.printc("Error in addSplineTool: no interactor found.", c='r')
            raise RuntimeError
        sw.On()
        sw.Initialize(sw.points.polydata())
        if sw.closed:
            sw.representation.ClosedLoopOn()
        sw.representation.SetRenderer(self.renderer)
        sw.representation.BuildRepresentation()
        sw.Render()
        if interactive:
            self.interactor.Start()
        else:
            self.interactor.Render()
        return sw


        return addons.addSplineTool(self, points, pc, ps, lc, ac, lw, closed, interactive)

    def addCutterTool(self, obj=None, mode='box', invert=False):
        """Create an interactive tool to cut away parts of a mesh or volume.

        :param str mode: either "box", "plane" or "sphere"
        :param bool invert: invert selection (inside-out)

        |cutter| |cutter.py|_
        """
        return addons.addCutterTool(obj, mode, invert)

    def addIcon(self, icon, pos=3, size=0.08):
        """Add an inset icon mesh into the same renderer.

        :param pos: icon position in the range [1-4] indicating one of the 4 corners,
                    or it can be a tuple (x,y) as a fraction of the renderer size.
        :param float size: size of the square inset.

        |icon| |icon.py|_
        """
        return addons.addIcon(icon, pos, size)

    def addGlobalAxes(self, axtype=None, c=None):
        """Draw axes on scene. Available axes types:

        :param int axtype:

            - 0,  no axes,
            - 1,  draw three gray grid walls
            - 2,  show cartesian axes from (0,0,0)
            - 3,  show positive range of cartesian axes from (0,0,0)
            - 4,  show a triad at bottom left
            - 5,  show a cube at bottom left
            - 6,  mark the corners of the bounding box
            - 7,  draw a 3D ruler at each side of the cartesian axes
            - 8,  show the ``vtkCubeAxesActor`` object
            - 9,  show the bounding box outLine
            - 10, show three circles representing the maximum bounding box
            - 11, show a large grid on the x-y plane
            - 12, show polar axes
            - 13, draw a simple ruler at the bottom of the window

        Axis type-1 can be fully customized by passing a dictionary ``axes=dict()``.

            :Example:

                .. code-block:: python

                    from vedo import Box, show
                    b = Box(pos=(0,0,0), length=80, width=90, height=70).alpha(0)

                    show(b, axes={ 'xtitle':'Some long variable [a.u.]',
                                   'numberOfDivisions':4,
                                   # ...
                                 }
                    )

        |customAxes1| |customAxes1.py|_  |customAxes2.py|_ |customAxes3.py|_

        |customIndividualAxes| |customIndividualAxes.py|_
        """
        addons.addGlobalAxes(axtype, c)
        return self

    def addLegendBox(self, **kwargs):
        """Add a legend to the top right"""
        acts = self.getMeshes()
        lb = addons.LegendBox(acts, **kwargs)
        self.add(lb)
        return self

    def _addSkybox(self, hdrfile):
        # many hdr files are at https://polyhaven.com/all
        if utils.vtkVersionIsAtLeast(9):

#            if self.skybox:
#                #already exists, skip.
#                return self

            reader = vtk.vtkHDRReader()
            # Check the image can be read.
            if not reader.CanReadFile(hdrfile):
                vedo.printc('Cannot read HDR file', hdrfile, c='r')
                return self
            reader.SetFileName(hdrfile)
            reader.Update()

            texture = vtk.vtkTexture()
            texture.SetColorModeToDirectScalars()
            texture.SetInputData(reader.GetOutput())

            # Convert to a cube map
            tcm = vtk.vtkEquirectangularToCubeMapTexture()
            tcm.SetInputTexture(texture)
            # Enable mipmapping to handle HDR image
            tcm.MipmapOn()
            tcm.InterpolateOn()

            self.renderer.SetEnvironmentTexture(tcm)
            self.renderer.UseImageBasedLightingOn()
            self.skybox = vtk.vtkSkybox()
            self.skybox.SetTexture(tcm)
            self.renderer.AddActor(self.skybox)
        else:
            vedo.printc("addSkyBox not supported in this VTK version. Skip.", c='r')
        return self

    def addRendererFrame(self, c=None, alpha=None, lw=None, pad=None):
        """
        Add a frame to the renderer subwindow

        Parameters
        ----------
        c : str, optional
            color name or index. The default is None.
        alpha : float, optional
            opacity. The default is None.
        lw : int, optional
            line width in pixels. The default is None.
        pad : float, optional
            padding space. The default is None.
        """
        self.frames = addons.addRendererFrame(self, c, alpha,lw, pad)
        return self


    def addHoverLegend(self,
                       at=0,
                       c=None,
                       pos='bottom-left',
                       font="Calco",
                       s=0.75,
                       bg='auto',
                       alpha=0.1,
                       precision=2,
                       maxlength=24,
                       useInfo=False,
        ):
        """Add a legend with 2D text which is triggered by hovering the mouse on an object.

        The created text object are stored in ``plotter.hoverLegends``.

        :param c: text color. If None then black or white is chosen automatically
        :param str pos: text positioning
        :param str font: text font
        :param float s: text size factor
        :param bg: background color of the 2D box containing the text
        :param float alpha: box transparency
        :param int precision: number of significant digits
        :param int maxlength: maximum number of characters per line
        :param bool useInfo: visualize the content of the ``obj.info`` attribute
        """
        hoverLegend = vedo.shapes.Text2D('', pos=pos, font=font, c=c, s=s, alpha=alpha, bg=bg)

        def _legfunc(evt):
            # helper function (png not pickable because of alpha channel in vtk9 ??)
            if not evt.actor or not self.renderer or at != evt.at:
                if hoverLegend._mapper.GetInput(): # clear and return
                    hoverLegend._mapper.SetInput('')
                    self.interactor.Render()
                return

            if useInfo:
                if hasattr(evt.actor, "info"):
                    t = str(evt.actor.info)
                else:
                    return
            else:
                t, tp = '', ''
                if evt.isMesh:
                    tp = "Mesh "
                elif evt.isPoints:
                    tp = "Points "
                # elif evt.isVolume: # todo -not working
                #     tp = "Volume "
                elif evt.isPicture:
                    tp = "Pict "
                elif evt.isAssembly:
                    tp = "Assembly "
                else:
                    return self

                if evt.isAssembly:
                    if not evt.actor.name:
                        t += f"Assembly object of {len(evt.actor.unpack())} parts\n"
                    else:
                        t += f"Assembly name: {evt.actor.name} ({len(evt.actor.unpack())} parts)"
                else:
                    if evt.actor.name:
                        t += f"{tp}name"
                        if evt.isPoints: t += '  '
                        if evt.isMesh: t += '  '
                        t += f": {evt.actor.name[:maxlength]}".ljust(maxlength)

                if evt.actor.filename:
                    if evt.actor.name: t +='\n'
                    t += f"{tp}filename: "
                    t += f"{os.path.basename(evt.actor.filename[-maxlength:])}".ljust(maxlength)
                    if not evt.actor.fileSize:
                        evt.actor.fileSize, evt.actor.created = vedo.io.fileInfo(evt.actor.filename) #BUG?
                    if evt.actor.fileSize:
                        t += "\n             : "
                        sz, created = evt.actor.fileSize, evt.actor.created
                        t += f"{created[4:-5]} ({sz})"

                if evt.isPoints:
                    indata = evt.actor.polydata(False)
                    if indata.GetNumberOfPoints():
                        t += f"\n#points/cells: {indata.GetNumberOfPoints()}"\
                             f" / {indata.GetNumberOfCells()}"
                    pdata = indata.GetPointData()
                    cdata = indata.GetCellData()
                    if pdata.GetScalars() and pdata.GetScalars().GetName():
                        t += f"\nPoint array  : {pdata.GetScalars().GetName()}"
                        if pdata.GetScalars().GetName() == evt.actor.mapper().GetArrayName():
                            t += " *"
                    if cdata.GetScalars() and cdata.GetScalars().GetName():
                        t += f"\nCell  array  : {cdata.GetScalars().GetName()}"
                        if cdata.GetScalars().GetName() == evt.actor.mapper().GetArrayName():
                            t += " *"

                if evt.isPicture:
                    t = f"{os.path.basename(evt.actor.filename[:maxlength+10])}".ljust(maxlength+10)
                    t += f"\nImage shape: {evt.actor.shape}"
                    pcol = vedo.colors.colorPicker(evt.picked2d, plotter=self)
                    t += f"\nPixel color: {vedo.colors.rgb2hex(pcol/255)} {pcol}"


            # change box color if needed in 'auto' mode
            if evt.isPoints and 'auto' in str(bg):
                actcol = evt.actor.GetProperty().GetColor()
                if hoverLegend._mapper.GetTextProperty().GetBackgroundColor() != actcol:
                    hoverLegend._mapper.GetTextProperty().SetBackgroundColor(actcol)

            # adapt to changes in bg color
            bgcol = self.renderers[at].GetBackground()
            _bgcol = c
            if _bgcol == None:  # automatic black or white
                _bgcol = (0.9, 0.9, 0.9)
                if sum(bgcol) > 1.5:
                    _bgcol = (0.1, 0.1, 0.1)
                if len(set(_bgcol).intersection(bgcol))<3:
                    hoverLegend.color(_bgcol)

            if hoverLegend._mapper.GetInput() != t:
                hoverLegend._mapper.SetInput(t)
                self.interactor.Render()

        self.add(hoverLegend, render=False, at=at)
        self.hoverLegends.append(hoverLegend)
        self.addCallback('MouseMove', _legfunc)
        return self


    def addCallback(self, eventName, func, priority=0.0, verbose=False):
        """Add a function to be executed while show() is active.
        Information about the event can be acquired with method ``getEvent()``.

        Return a unique id for the callback.

        The callback function (see example below) exposes a dictionary
        with the following information:

            - ``name``: event name,
            - ``id``: event unique identifier,
            - ``priority``: event priority (float),
            - ``interactor``: the interactor object,
            - ``at``: renderer nr. where the event occured
            - ``actor``: object picked by the mouse
            - ``picked3d``: point picked in world coordinates
            - ``keyPressed``: key pressed as string
            - ``picked2d``: screen coords of the mouse pointer
            - ``delta2d``: shift wrt previous position (to calculate speed, direction)
            - ``delta3d``: ...same but in 3D world coords
            - ``angle2d``: angle of mouse movement on screen
            - ``speed2d``: speed of mouse movement on screen
            - ``speed3d``: speed of picked point in world coordinates
            - ``isPoints``: True if of class
            - ``isMesh``: True if of class
            - ``isAssembly``: True if of class
            - ``isVolume``: True if of class Volume
            - ``isPicture``: True if of class

        Frequently used events are:

            - KeyPress, KeyRelease: listen to keyboard events
            - LeftButtonPress, LeftButtonRelease: listen to mouse clicks
            - MiddleButtonPress, MiddleButtonRelease
            - RightButtonPress, RightButtonRelease
            - MouseMove: listen to mouse pointer changing position
            - MouseWheelForward, MouseWheelBackward
            - Enter, Leave: listen to mouse entering or leaving the window
            - Pick, StartPick, EndPick: listen to object picking
            - ResetCamera, ResetCameraClippingRange
            - Error, Warning
            - Char
            - Timer

        Check the complete list of events here:
            https://vtk.org/doc/nightly/html/classvtkCommand.html

        :Example:

            .. code-block:: python

                from vedo import *

                def func(evt): # called every time the mouse moves
                    # evt is a dotted dictionary
                    if not evt.actor:
                        return  # no hit, return
                    print("point coords =", evt.picked3d)
                    # print("full event dump:", evt)

                elli = Ellipsoid()
                plt = show(elli, axes=1, interactive=False)
                plt.addCallback('MouseMove', func)
                interactive()
        """
        if not self.interactor:
            return None

        # as vtk names are ugly and difficult to remember:
        ln = eventName.lower()
        if "click" in ln or "button" in ln:
            eventName="LeftButtonPress"
            if "right" in ln:
                eventName="RightButtonPress"
            elif "mid" in ln:
                eventName="MiddleButtonPress"
            if "release" in ln:
                # eventName = eventName.replace("Press","Release") # vtk bug
                eventName="EndInteraction"
        else:
            if "key" in ln:
                if 'release' in ln:
                    eventName="KeyRelease"
                else:
                    eventName="KeyPress"

        if ("mouse" in ln and "mov" in ln) or "over" in ln:
            eventName="MouseMove"
        if "timer" in ln:
            eventName="Timer"

        if not eventName.endswith('Event'):
            eventName += 'Event'

        # print(eventName)

        def _func_wrap(iren, ename):
            x, y = self.interactor.GetEventPosition()
            self.renderer = self.interactor.FindPokedRenderer(x, y)
            if not self.picker:
                self.picker = vtk.vtkPropPicker()
            self.picked2d = (x,y)
            self.picker.PickProp(x, y, self.renderer)
            xp, yp = self.interactor.GetLastEventPosition()
            actor = self.picker.GetProp3D()
            delta3d = np.array([0,0,0])
            if actor:
                picked3d = np.array(self.picker.GetPickPosition())
                if isinstance(actor, vedo.base.Base3DProp): # needed!
                    if actor.picked3d is not None:
                        delta3d = picked3d - actor.picked3d
                actor.picked3d = picked3d
            else:
                picked3d = None

            if not actor: # try 2D
                actor = self.picker.GetActor2D()

            dx, dy = x-xp, y-yp

            event_dict = utils.dotdict({
                "name": ename,
                "title": self.title, # window title, can be used as an id for the Plotter
                "id": cid,
                "priority": priority,
                "at": self.renderers.index(self.renderer),
                "actor": actor,
                "picked3d": picked3d,
                "keyPressed": self.interactor.GetKeySym(),
                "picked2d": (x,y),
                "delta2d": (dx, dy),
                "angle2d": np.arctan2(dy,dx),
                "speed2d": np.sqrt(dx*dx+dy*dy),
                "delta3d": delta3d,
                "speed3d": np.sqrt(np.dot(delta3d,delta3d)),
                "isPoints":   isinstance(actor, vedo.Points),
                "isMesh":     isinstance(actor, vedo.Mesh),
                "isAssembly": isinstance(actor, vedo.Assembly),
                "isVolume":   isinstance(actor, vedo.Volume),
                "isPicture":  isinstance(actor, vedo.Picture),
                "isActor2D":  isinstance(actor, vtk.vtkActor2D),
            })
            func(event_dict)
            return   ## _func_wrap

        if self._timer_event_id is not None:
            # lets remove the existing allowInteraction callback
            #  to avoid interference with the user one
            self.interactor.RemoveObserver(self._timer_event_id)
            self._timer_event_id = None

        cid = self.interactor.AddObserver(eventName, _func_wrap, priority)
        if verbose:
            vedo.printc('addCallback(): registering event:', eventName, 'with id =', cid)
        return cid

    def removeCallback(self, cid):
        """Remove a callback function by its id
        or a whole category of callbacks by their name.

        :param int,str cid: unique id of the callback.
            If an event name is passed all callbacks of that type are removed
        """
        if self.interactor:
            if isinstance(cid, str):
                # as vtk names are ugly and difficult to remember:
                ln = cid.lower()
                if "click" in ln or "button" in ln:
                    cid="LeftButtonPress"
                    if "right" in ln:
                        cid="RightButtonPress"
                    elif "mid" in ln:
                        cid="MiddleButtonPress"
                    if "release" in ln:
                        cid.replace("Press","Release")
                else:
                    if "key" in ln:
                        if 'release' in ln:
                            cid="KeyRelease"
                        else:
                            cid="KeyPress"
                if ("mouse" in ln and "mov" in ln) or "over" in ln:
                    cid="MouseMove"
                if "timer" in ln:
                    cid="Timer"
                if not cid.endswith('Event'):
                    cid += 'Event'
                self.interactor.RemoveObservers(cid)
            else:
                self.interactor.RemoveObserver(cid)
        return self

    def timerCallback(self, action, timerId=None, dt=10, oneShot=False):
        """
        Activate or destroy an existing Timer Event callback.

        Parameters
        ----------
        action : str
            Either "create" or "destroy".
        timerId : int
            When destroying the timer, the ID of the timer as returned when created.
        dt : int
            time in milliseconds between each repeated call
        oneShot: bool
            create a one shot timer of prescribed duration instead of a repeating one.
        """
        if action == "create":
            if oneShot:
                timer_id = self.interactor.CreateOneShotTimer(dt)
            else:
                timer_id = self.interactor.CreateRepeatingTimer(dt)
            return timer_id
        elif action == "destroy":
            if timerId is not None:
                self.interactor.DestroyTimer(timerId)
        else:
            vedo.printc("Error in plotter.timer(). Cannot understand action:", action, c='r')
            vedo.printc("                          allowed actions: [create, destroy]", action, c='r')
        return self


    def computeWorldPosition(self, pos2d, at=0, objs=(), bounds=(),
                             offset=None, pixeltol=None, worldtol=None):
        """
        Transform a 2D point on the screen into a 3D point inside the rendering scene.

        Parameters
        ----------
        pos2d : list
            2D screen coordinates point.
        at : int, optional
            renderer number. The default is 0.
        objs : list, optional
            list of Mesh objects to project the point onto. The default is ().
        bounds : list, optional
            specify a bounding box as [xmin,xmax, ymin,ymax, zmin,zmax]. The default is ().
        offset : float, optional
            specify an offset value. The default is None (will use system defaults).
        pixeltol : int, optional
            screen tolerance in pixels. The default is None (will use system defaults).
        worldtol : float, optional
            world coordinates tolerance. The default is None (will use system defaults).

        Returns
        -------
        numpy array
            the point in 3D world coordinates.
        """
        renderer = self.renderers[at]
        if not objs:
            pp = vtk.vtkFocalPlanePointPlacer()
        else:
            pp = vtk.vtkPolygonalSurfacePointPlacer()
            for ob in objs:
                pp.AddProp(ob)

        if len(bounds)==6:
            pp.SetPointBounds(bounds)
        if pixeltol:
            pp.SetPixelTolerance(pixeltol)
        if worldtol:
            pp.SetWorldTolerance(worldtol)
        if offset:
            pp.SetOffset(offset)

        worldPos = [0,0,0]
        worldOrient = [0,0,0, 0,0,0, 0,0,0]
        pp.ComputeWorldPosition(renderer, pos2d, worldPos, worldOrient)
        # validw = pp.ValidateWorldPosition(worldPos, worldOrient)
        # validd = pp.ValidateDisplayPosition(renderer, pos2d)
        # print(validd, validw, worldOrient)
        return np.array(worldPos)


    def _scan_input(self, wannabeacts):

        if not utils.isSequence(wannabeacts):
            wannabeacts = [wannabeacts]

        scannedacts = []
        for a in wannabeacts:  # scan content of list

            if a is None:
                pass

            elif isinstance(a, vtk.vtkActor):
                scannedacts.append(a)
                if isinstance(a, vedo.base.BaseActor):
                    if a.trail and a.trail not in self.actors:
                        scannedacts.append(a.trail)
                    if a.shadow and a.shadow not in self.actors:
                        scannedacts.append(a.shadow)
                    if a._caption and a._caption not in self.actors:
                        scannedacts.append(a._caption)

            elif isinstance(a, vtk.vtkActor2D):
                scannedacts.append(a)

            elif isinstance(a, vtk.vtkAssembly):
                scannedacts.append(a)
                if a.trail and a.trail not in self.actors:
                    scannedacts.append(a.trail)

            elif isinstance(a, (vedo.Volume, vedo.VolumeSlice)):
                scannedacts.append(a)

            elif isinstance(a, vtk.vtkImageData):
                scannedacts.append(vedo.Volume(a))

            elif isinstance(a, vedo.TetMesh):
                # check ugrid is all made of tets
                ugrid = a.inputdata()
                uarr = ugrid.GetCellTypesArray()
                celltypes = np.unique(utils.vtk2numpy(uarr))
                ncelltypes = len(celltypes)
                if ncelltypes > 1 or (ncelltypes==1 and celltypes[0]!=10):
                    scannedacts.append(a.tomesh())
                else:
                    if not ugrid.GetPointData().GetScalars():
                        if not ugrid.GetCellData().GetScalars():
                            #add dummy array for vtkProjectedTetrahedraMapper to work:
                            a.addCellArray(np.ones(a.NCells()), 'DummyOneArray')
                    scannedacts.append(a)

            elif isinstance(a, vedo.UGrid):
                scannedacts.append(a.tomesh())

            elif isinstance(a, vtk.vtkVolume): # order matters! dont move above TetMesh
                vvol = vedo.Volume(a.GetMapper().GetInput())
                vprop = vtk.vtkVolumeProperty()
                vprop.DeepCopy(a.GetProperty())
                vvol.SetProperty(vprop)
                scannedacts.append(vvol)

            elif isinstance(a, str):
                # assume a 2D comment was given
                changed = False  # check if one already exists so to just update text
                if self.renderer: # might be jupyter
                    acs = self.renderer.GetActors2D()
                    acs.InitTraversal()
                    for i in range(acs.GetNumberOfItems()):
                        act = acs.GetNextItem()
                        if isinstance(act, vedo.shapes.Text2D):
                            aposx, aposy = act.GetPosition()
                            if aposx<0.01 and aposy>0.99: # "top-left"
                                act.text(a)  # update content! no appending nada
                                changed = True
                                break
                    if not changed:
                        out = vedo.shapes.Text2D(a) # append a new one
                        scannedacts.append(out)

            elif isinstance(a, vtk.vtkImageActor):
                scannedacts.append(a)

            elif isinstance(a, vtk.vtkBillboardTextActor3D):
                scannedacts.append(a)

            elif isinstance(a, vtk.vtkLight):
                self.renderer.AddLight(a)

            elif isinstance(a, vtk.vtkMultiBlockDataSet):
                for i in range(a.GetNumberOfBlocks()):
                    b =  a.GetBlock(i)
                    if isinstance(b, vtk.vtkPolyData):
                        scannedacts.append(vedo.Mesh(b))
                    elif isinstance(b, vtk.vtkImageData):
                        scannedacts.append(vedo.Volume(b))

            elif "PolyData" in str(type(a)):  # assume a pyvista obj
                scannedacts.append(vedo.Mesh(a))

            elif "dolfin" in str(type(a)):  # assume a dolfin.Mesh object
                scannedacts.append(vedo.dolfin.MeshActor(a))

            elif "trimesh" in str(type(a)):
                scannedacts.append(utils.trimesh2vedo(a))

            elif "meshlab" in str(type(a)):
                if "MeshSet" in str(type(a)):
                    for i in range(a.number_meshes()):
                        if a.mesh_id_exists(i):
                            scannedacts.append(vedo.Mesh(utils._meshlab2vedo(a.mesh(i))))
                else:
                    scannedacts.append(vedo.Mesh(utils._meshlab2vedo(a)))

            else:
                vedo.printc("Error: cannot understand input in show():", type(a), c='r')
        return scannedacts


    def show(self, *actors,
                    at=None,
                    axes=None,
                    resetcam=None,
                    zoom=False,
                    interactive=None,
                    viewup="",
                    azimuth=0,
                    elevation=0,
                    roll=0,
                    camera=None,
                    interactorStyle=0,
                    mode=None,
                    rate=None,
                    bg=None,
                    bg2=None,
                    size=None,
                    title=None,
                    q=False,
        ):
        """
        Render a list of actors.

        If filename is given, its type is guessed based on its extension.
        Supported formats are:
        `vtu, vts, vtp, ply, obj, stl, 3ds, xml, neutral, gmsh, pcd, xyz, txt, byu,
        tif, slc, vti, mhd, png, jpg`.
        Otherwise it will be interpreted as a comment to appear on the top-left of the window.

        :param int at: number of the renderer to plot to, if more than one exists
        :param list shape: Number of sub-render windows inside of the main window.
            Specify two across with ``shape=(2, 1)`` and a two by two grid with ``shape=(2, 2)``.
            By default there is only one renderer.
            Can also accept a shape as string descriptor. E.g.

            - shape="3|1" means 3 plots on the left and 1 on the right,
            - shape="4/2" means 4 plots on top of 2 at bottom.

        :param int axes: set the type of axes to be shown

            - 0,  no axes
            - 1,  draw three customizable gray grid walls
            - 2,  show cartesian axes from (0,0,0)
            - 3,  show positive range of cartesian axes from (0,0,0)
            - 4,  show a triad at bottom left
            - 5,  show a cube at bottom left
            - 6,  mark the corners of the bounding box
            - 7,  draw a 3D ruler at each side of the cartesian axes
            - 8,  show the ``vtkCubeAxesActor`` object
            - 9,  show the bounding box outLine
            - 10, show three circles representing the maximum bounding box
            - 11, show a large grid on the x-y plane (use with zoom=8)
            - 12, show polar axes
            - 13, draw a simple ruler at the bottom of the window

        :param float azimuth/elevation/roll:  move camera accordingly
        :param str viewup:  either ['x', 'y', 'z'] to set vertical direction
        :param bool resetcam:  re-adjust camera position to fit objects
        :param dict camera: Camera parameters can further be specified with a dictionary assigned
            to the ``camera`` keyword (E.g. `show(camera={'pos':(1,2,3), 'thickness':1000,})`)

            - pos, `(list)`,  the position of the camera in world coordinates
            - focalPoint `(list)`, the focal point of the camera in world coordinates
            - viewup `(list)`, the view up direction vector for the camera
            - distance `(float)`, set the focal point to the specified distance from the camera position.
            - clippingRange `(float)`, distance of the near and far clipping planes along
                the direction of projection.

            - parallelScale `(float)`,
                scaling used for a parallel projection, i.e. the height of the viewport
                in world-coordinate distances. The default is 1. Note that the "scale"
                parameter works as an "inverse scale", larger numbers produce smaller images.
                This method has no effect in perspective projection mode.

            - thickness `(float)`,
                set the distance between clipping planes. This method adjusts the far clipping
                plane to be set a distance 'thickness' beyond the near clipping plane.

            - viewAngle `(float)`,
                the camera view angle, which is the angular height of the camera view
                measured in degrees. The default angle is 30 degrees.
                This method has no effect in parallel projection mode.
                The formula for setting the angle up for perfect perspective viewing is:
                angle = 2*atan((h/2)/d) where h is the height of the RenderWindow
                (measured by holding a ruler up to your screen) and d is the distance from your
                eyes to the screen.

        :param bool interactive:  pause and interact with window (True)
            or continue execution (False)

        :param float rate:  maximum rate of `show()` in Hertz
        :param int,str mode: set the type of interaction
            - 0 = TrackballCamera [default]
            - 1 = TrackballActor
            - 2 = JoystickCamera
            - 3 = JoystickActor
            - 4 = Flight
            - 5 = RubberBand2D
            - 6 = RubberBand3D
            - 7 = RubberBandZoom
            - 8 = Context
            - 9 = 3D
            - 10 = Terrain
            - 11 = Unicam
            - 12 = Image

        :param bool q:  force program to quit after `show()` command returns.
        """
        if self.wxWidget:
            return self

        if title is not None:
            self.title = title

        if mode is not None: ### interactorStyle will disappear in later releases!
            interactorStyle = mode

        if size is not None:
            self.size = size
            if self.size[0] == 'f':  # full screen
                self.size = 'fullscreen'
                self.window.SetFullScreen(True)
                self.window.BordersOn()
            else:
                self.window.SetSize(int(self.size[0]), int(self.size[1]))

        if at is not None and len(self.renderers)>at:
            self.renderer = self.renderers[at]

        if not settings.notebookBackend:
            if str(bg).endswith(".hdr"):
                self._addSkybox(bg)
            else:
                if bg is not None:
                    self.backgrcol = vedo.getColor(bg)
                    self.renderer.SetBackground(self.backgrcol)
                if bg2 is not None:
                    self.renderer.GradientBackgroundOn()
                    self.renderer.SetBackground2(vedo.getColor(bg2))

        if axes is not None:
            if isinstance(axes, vedo.Assembly): # user passing show(..., axes=myaxes)
                actors = list(actors)
                actors.append(axes) # move it into the list of normal things to show
                axes = 0
            self.axes = axes

        if self.offscreen:
            interactive = False
            self.interactive = False

        if camera is not None:
            self.resetcam = False
            if isinstance(camera, vtk.vtkCamera):
                self.camera = camera

        if resetcam is not None:
            self.resetcam = resetcam

        if len(actors) == 0:
            actors = None
        elif len(actors) == 1:
            actors = actors[0]
        else:
            actors = utils.flatten(actors)

        if actors is not None:
            self.actors = []
            actors2show = self._scan_input(actors)
            for a in actors2show:
                if a not in self.actors:
                    self.actors.append(a)
        else:
            actors2show = self._scan_input(self.actors)
            self.actors = list(actors2show)


        # Backend ###############################################################
        if settings.notebookBackend:
            if settings.notebookBackend not in ['panel', '2d', 'ipyvtk']:
                return backends.getNotebookBackend(actors2show, zoom, viewup)
        #########################################################################
        # check if the widow needs to be closed (ESC button was hit)
        if self.escaped:
            if not self.window:
                return self # do nothing, just return self (was already closed)
            for r in self.renderers:
                r.RemoveAllObservers()
            self.camera.RemoveAllObservers()
            self.closeWindow()
            #reset some important defaults in case vedo is not reloaded..
            settings.defaultFont = 'Normografo'
            settings.interactorStyle = None
            settings.immediateRendering = True
            settings.multiSamples = 8
            settings.xtitle = "x"
            settings.ytitle = "y"
            settings.ztitle = "z"
            return self

        if interactive is not None:
            self.interactive = interactive

        if self.interactor:
            if not self.interactor.GetInitialized():
                self.interactor.Initialize()

        if at is None and len(self.renderers) > 1:
            # in case of multiple renderers a call to show w/o specifying
            # at which renderer will just render the whole thing and return
            if self.interactor:
                if zoom:
                    self.camera.Zoom(zoom)
                self.window.Render()
                self.window.SetWindowName(self.title)
                if self.interactive:
                    self.interactor.Start()
                return self ###############

        if at is None:
            at = 0

        if at < len(self.renderers):
            self.renderer = self.renderers[at]
        else:
            if settings.notebookBackend:
                vedo.printc("Error in show(): multiple renderings not supported in notebooks.", c='r')
            else:
                vedo.printc("Error in show(): wrong renderer index", at, c='r')
            return self

        if self.qtWidget is not None:
            self.qtWidget.GetRenderWindow().AddRenderer(self.renderer)

        if not self.camera:
            self.camera = self.renderer.GetActiveCamera()

        self.camera.SetParallelProjection(settings.useParallelProjection)

        if self.sharecam:
            for r in self.renderers:
                r.SetActiveCamera(self.camera)

        if len(self.renderers) == 1:
            self.renderer.SetActiveCamera(self.camera)

        if settings.vtk_version[0] == 9 and "Darwin" in settings.sys_platform:
            for a in self.actors:
                if isinstance(a, vtk.vtkVolume):
                    self.window.SetMultiSamples(0) # to fix mac OSX BUG vtk9
                    break

        # rendering
        for ia in actors2show:  # add the actors that are not already in scene
            if ia:
                if isinstance(ia, vtk.vtkVolume):
                    self.renderer.AddVolume(ia)
                else:
                    self.renderer.AddActor(ia)

                if hasattr(ia, '_set2actcam') and ia._set2actcam:
                    ia.SetCamera(self.camera)  # used by mesh.followCamera()

                if hasattr(ia, 'renderedAt'):
                    ia.renderedAt.add(at)

                if hasattr(ia, 'scalarbar') and ia.scalarbar:
                    self.renderer.AddActor(ia.scalarbar)
                    # fix gray color labels and title to white or black
                    if isinstance(ia.scalarbar, vtk.vtkScalarBarActor):
                        ltc = np.array(ia.scalarbar.GetLabelTextProperty().GetColor())
                        if np.linalg.norm(ltc-(.5,.5,.5))/3 < 0.05:
                            c = (0.9, 0.9, 0.9)
                            if np.sum(self.renderer.GetBackground()) > 1.5:
                                c = (0.1, 0.1, 0.1)
                            ia.scalarbar.GetLabelTextProperty().SetColor(c)
                            ia.scalarbar.GetTitleTextProperty().SetColor(c)
                    if ia.scalarbar not in self.scalarbars:
                        self.scalarbars.append(ia.scalarbar)

                if (hasattr(ia, 'flagText')
                    and self.interactor
                    and not self.offscreen
                    and not (settings.vtk_version[0] == 9 and "Linux" in settings.sys_platform)
                    ):
                    #check balloons
                    # Linux vtk9 is bugged
                    if ia.flagText:
                        if not self.flagWidget: # Create widget on the fly
                            self._flagRep = vtk.vtkBalloonRepresentation()
                            self._flagRep.SetBalloonLayoutToImageRight()
                            breppr = self._flagRep.GetTextProperty()
                            breppr.SetFontFamily(vtk.VTK_FONT_FILE)
                            breppr.SetFontFile(utils.getFontPath(settings.flagFont))
                            breppr.SetFontSize(settings.flagFontSize)
                            breppr.SetColor(vedo.getColor(settings.flagColor))
                            breppr.SetBackgroundColor(vedo.getColor(settings.flagBackgroundColor))
                            breppr.SetShadow(settings.flagShadow)
                            breppr.SetJustification(settings.flagJustification)
                            breppr.UseTightBoundingBoxOn()
                            if settings.flagAngle:
                                breppr.SetOrientation(settings.flagAngle)
                                breppr.SetBackgroundOpacity(0)
                            self.flagWidget = vtk.vtkBalloonWidget()
                            self.flagWidget.SetTimerDuration(settings.flagDelay)
                            self.flagWidget.ManagesCursorOff()
                            self.flagWidget.SetRepresentation(self._flagRep)
                            self.flagWidget.SetInteractor(self.interactor)
                            self.widgets.append(self.flagWidget)
                        bst = self.flagWidget.GetBalloonString(ia)
                        if bst:
                            if bst != ia.flagText:
                                self.flagWidget.UpdateBalloonString(ia, ia.flagText)
                        else:
                            self.flagWidget.AddBalloon(ia, ia.flagText)
                    if ia.flagText is False and self.flagWidget:
                        self.flagWidget.RemoveBalloon(ia)

        # remove the ones that are not in actors2show (and their scalarbar if any)
        for ia in self.getMeshes(at, includeNonPickables=True) + self.getVolumes(at, includeNonPickables=True):
            if ia not in actors2show:
                if isinstance(ia, vtk.vtkSkybox):
                    continue
                self.renderer.RemoveActor(ia)
                if hasattr(ia, 'scalarbar') and ia.scalarbar:
                    if isinstance(ia.scalarbar, vtk.vtkActor):
                        self.renderer.RemoveActor(ia.scalarbar)
                    elif isinstance(ia.scalarbar, vedo.Assembly):
                        for a in ia.scalarbar.unpack():
                            self.renderer.RemoveActor(a)
                if hasattr(ia, 'renderedAt'):
                    ia.renderedAt.discard(at)


        if self.axes is not None:
            if viewup != "2d" or self.axes in [1, 8] or isinstance(self.axes, dict):
                addons.addGlobalAxes(self.axes)

        # panel #################################################################
        if settings.notebookBackend in ["panel","ipyvtk"]:
            return backends.getNotebookBackend(0, 0, 0)
        #########################################################################

        if self.resetcam:
            self.renderer.ResetCamera()

        if len(self.renderers) > 1:
            self.frames = self.addRendererFrame()

        if self.flagWidget:
            self.flagWidget.EnabledOn()

        if zoom:
            self.camera.Zoom(zoom)
        if elevation:
            self.camera.Elevation(elevation)
        if azimuth:
            self.camera.Azimuth(azimuth)
        if roll:
            self.camera.Roll(roll)

        if self._first_viewup and len(viewup):
            self._first_viewup = False # gets executed only once
            if viewup == "x":
                self.camera.SetViewUp([1, 0.001, 0])
            elif viewup == "y":
                self.camera.SetViewUp([0.001, 1, 0])
            elif viewup == "z":
                b =  self.renderer.ComputeVisiblePropBounds()
                self.camera.SetViewUp([0, 0.001, 1])
                cm = [(b[1]+b[0])/2, (b[3]+b[2])/2, (b[5]+b[4])/2]
                sz = np.array([(b[1]-b[0])*0.7, -(b[3]-b[2])*1.0, (b[5]-b[4])*1.2])
                self.camera.SetPosition(cm+2*sz)
            elif viewup == "2d":
                interactorStyle = 12

        if isinstance(camera, dict):
            camera = dict(camera) # make a copy so input is not emptied by pop()
            cm_pos = camera.pop("pos", None)
            cm_focalPoint = camera.pop("focalPoint", None)
            cm_viewup = camera.pop("viewup", None)
            cm_distance = camera.pop("distance", None)
            cm_clippingRange = camera.pop("clippingRange", None)
            cm_parallelScale = camera.pop("parallelScale", None)
            cm_thickness = camera.pop("thickness", None)
            cm_viewAngle = camera.pop("viewAngle", None)
            if len(camera.keys()):
                vedo.printc("Warning in show(cam=...), key(s) not recognized:",
                       *(camera.keys()), c='y')
            if cm_pos is not None: self.camera.SetPosition(cm_pos)
            if cm_focalPoint is not None: self.camera.SetFocalPoint(cm_focalPoint)
            if cm_viewup is not None: self.camera.SetViewUp(cm_viewup)
            if cm_distance is not None: self.camera.SetDistance(cm_distance)
            if cm_clippingRange is not None: self.camera.SetClippingRange(cm_clippingRange)
            if cm_parallelScale is not None: self.camera.SetParallelScale(cm_parallelScale)
            if cm_thickness is not None: self.camera.SetThickness(cm_thickness)
            if cm_viewAngle is not None: self.camera.SetViewAngle(cm_viewAngle)


        self.renderer.ResetCameraClippingRange()
        if settings.immediateRendering:
            self.window.Render() ##################################################### <----Render
        self.window.SetWindowName(self.title)

        # 2d ####################################################################
        if settings.notebookBackend == "2d":
            return backends.getNotebookBackend(0, 0, 0)
        #########################################################################

        if settings.allowInteraction and not self.offscreen:
            self.allowInteraction()

        # Set the style of interaction
        # see https://vtk.org/doc/nightly/html/classvtkInteractorStyle.html
        if settings.interactorStyle is not None:
            interactorStyle = settings.interactorStyle
        if interactorStyle == 0 or interactorStyle == "TrackballCamera":
            #csty = self.interactor.GetInteractorStyle().GetCurrentStyle().GetClassName()
            #if "TrackballCamera" not in csty:
            # this causes problems (when pressing 3 eg) :
            if self.qtWidget:
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            # pass
        elif interactorStyle == 1 or interactorStyle == "TrackballActor":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
        elif interactorStyle == 2 or interactorStyle == "JoystickCamera":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleJoystickCamera())
        elif interactorStyle == 3 or interactorStyle == "JoystickActor":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleJoystickActor())
        elif interactorStyle == 4 or interactorStyle == "Flight":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleFlight())
        elif interactorStyle == 5 or interactorStyle == "RubberBand2D":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBand2D())
        elif interactorStyle == 6 or interactorStyle == "RubberBand3D":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBand3D())
        elif interactorStyle == 7 or interactorStyle == "RubberBandZoom":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandZoom())
        elif interactorStyle == 8 or interactorStyle == "Context":
            self.interactor.SetInteractorStyle(vtk.vtkContextInteractorStyle())
        elif interactorStyle == 9 or interactorStyle == "3D":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyle3D())
        elif interactorStyle ==10 or interactorStyle == "Terrain":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTerrain())
        elif interactorStyle ==11 or interactorStyle == "Unicam":
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleUnicam())
        elif interactorStyle ==12 or interactorStyle == "Image" or interactorStyle == "image":
            astyle = vtk.vtkInteractorStyleImage()
            astyle.SetInteractionModeToImage3D()
            self.interactor.SetInteractorStyle(astyle)

        if self.interactor and self.interactive:
            self.interactor.Start()

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

        if q:  # exit python
            sys.exit(0)

        return self


    def addInset(self, *actors, **options):
        """Add a draggable inset space into a renderer.

        :param int at: specify the renderer number
        :param pos: icon position in the range [1-4] indicating one of the 4 corners,
                    or it can be a tuple (x,y) as a fraction of the renderer size.

        :param float size: size of the square inset.
        :param bool draggable: if True the subrenderer space can be dragged around.
        :param c: color of the inset frame when dragged

        |inset| |inset.py|_
        """
        if not self.interactor:
            return None
        pos = options.pop("pos", 0)
        size = options.pop("size", 0.1)
        c = options.pop("c", 'lb')
        at = options.pop("at", None)
        draggable = options.pop("draggable", True)

        if not self.renderer:
            vedo.printc("Use showInset() after first rendering the scene.", c='y')
            save_int = self.interactive
            self.show(interactive=0)
            self.interactive = save_int
        widget = vtk.vtkOrientationMarkerWidget()
        r,g,b = vedo.getColor(c)
        widget.SetOutlineColor(r,g,b)
        if len(actors)==1:
            widget.SetOrientationMarker(actors[0])
        else:
            widget.SetOrientationMarker(vedo.Assembly(actors))

        widget.SetInteractor(self.interactor)

        if utils.isSequence(pos):
            widget.SetViewport(pos[0]-size, pos[1]-size, pos[0]+size, pos[1]+size)
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
        self.widgets.append(widget)
        return widget


    def clear(self, actors=None, at=None):
        """Delete specified list of actors, by default delete all."""
        if at is not None:
            self.renderer = self.renderers[at]
        if actors is None:
            self.renderer.RemoveAllViewProps()
            self.actors = []
            self.scalarbars = []
            self.sliders = []
            self.buttons = []
            self.widgets = []
            self.scalarbars = []
            return self

        if not utils.isSequence(actors):
            actors = [actors]
        if len(actors):
            for a in actors:
                self.remove(a)
        elif self.renderer:
            self.actors = []
            for a in self.getMeshes():
                self.renderer.RemoveActor(a)
            for a in self.getVolumes():
                self.renderer.RemoveVolume(a)
            for s in self.sliders:
                s.EnabledOff()
            for b in self.buttons:
                self.renderer.RemoveActor(b)
            for w in self.widgets:
                w.EnabledOff()
            for a in self.scalarbars:
                self.renderer.RemoveActor(a)
            self.scalarbars = []


    def closeWindow(self):
        """Close the current or the input rendering window."""
        if hasattr(self, 'window') and self.window:
            self.window.Finalize()
            if hasattr(self, 'interactor') and self.interactor:
                self.interactor.ExitCallback()
                self.interactor.TerminateApp()
                #del self.window
                #del self.interactor
                self.window = None
                self.interactor = None
        return self

    def close(self):
        """Close the Plotter instance and release resources."""
        #self.clear()
        if hasattr(self, 'interactor') and self.interactor:
            self.interactor.ExitCallback()
        for r in self.renderers:
            r.RemoveAllObservers()
        self.camera.RemoveAllObservers()
        self.closeWindow()
        self.actors = []
        #reset some important defaults in case vedo is not reloaded..
        settings.defaultFont = 'Normografo'
        settings.interactorStyle = None
        settings.immediateRendering = True
        settings.multiSamples = 8
        settings.xtitle = "x"
        settings.ytitle = "y"
        settings.ztitle = "z"
        if settings.plotter_instance in settings.plotter_instances:
             settings.plotter_instances.remove(settings.plotter_instance)
        settings.plotter_instance = None

    def screenshot(self, filename='screenshot.png', scale=None, returnNumpy=False):
        """Take a screenshot of the Plotter window.

        :param int scale: set image magnification
        :param bool returnNumpy: return a numpy array of the image
        """
        retval = vedo.io.screenshot(filename, scale, returnNumpy)
        return retval

    def topicture(self, scale=None):
        """Generate a Picture object from the current rendering window.

        :param int scale: set image magnification
        """
        if scale is None:
            scale = settings.screeshotScale

        if settings.screeshotLargeImage:
           w2if = vtk.vtkRenderLargeImage()
           w2if.SetInput(settings.plotter_instance.renderer)
           w2if.SetMagnification(scale)
        else:
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(settings.plotter_instance.window)
            if hasattr(w2if, 'SetScale'):
                w2if.SetScale(scale, scale)
            if settings.screenshotTransparentBackground:
                w2if.SetInputBufferTypeToRGBA()
            w2if.ReadFrontBufferOff()  # read from the back buffer
        w2if.Update()
        return vedo.picture.Picture(w2if.GetOutput())

    def export(self, filename='scene.npz', binary=False):
        """Export scene to file to HTML, X3D or Numpy file."""
        vedo.io.exportWindow(filename, binary=binary)
        return self


    #######################################################################
    def _mouseleft(self, iren, event):

        x, y = iren.GetEventPosition()

        renderer = iren.FindPokedRenderer(x, y)
        self.renderer = renderer

        picker = vtk.vtkPropPicker()
        picker.PickProp(x, y, renderer)

        clickedActor = picker.GetActor()

        # print('_mouseleft mouse at', x, y)
        # print("picked Volume:",   [picker.GetVolume()])
        # print("picked Actor2D:",  [picker.GetActor2D()])
        # print("picked Assembly:", [picker.GetAssembly()])
        # print("picked Prop3D:",   [picker.GetProp3D()])

        # check if any button objects are clicked
        clickedActor2D = picker.GetActor2D()
        if clickedActor2D:
            for bt in self.buttons:
                if clickedActor2D == bt.actor:
                    bt.function()
                    break

        if not clickedActor:
            clickedActor = picker.GetAssembly()

        if not clickedActor:
            clickedActor = picker.GetProp3D()

        if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
            return

        self.picked3d = picker.GetPickPosition()
        self.picked2d = np.array([x,y])

        if not clickedActor:
            return

        self.justremoved = None

        self.clickedActor = clickedActor
        if hasattr(clickedActor, 'picked3d'): # might be not a vedo obj
            clickedActor.picked3d = picker.GetPickPosition()

        if self.mouseLeftClickFunction:
            self.mouseLeftClickFunction(clickedActor)


    #######################################################################
    def _mouseright(self, iren, event):

        x, y = iren.GetEventPosition()
        # print('_mouseright mouse at', x, y)

        renderer = iren.FindPokedRenderer(x, y)
        self.renderer = renderer

        picker = vtk.vtkPropPicker()
        picker.PickProp(x, y, renderer)
        clickedActor = picker.GetActor()

        # check if any button objects were created
        clickedActor2D = picker.GetActor2D()
        if clickedActor2D:
            for bt in self.buttons:
                if clickedActor2D == bt.actor:
                    bt.function()
                    break

        if not clickedActor:
            clickedActor = picker.GetAssembly()

        if not clickedActor:
            clickedActor = picker.GetProp3D()

        if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
            return
        self.picked3d = picker.GetPickPosition()
        self.picked2d = np.array([x,y])
        self.clickedActor = clickedActor

        if self.mouseRightClickFunction:
            self.mouseRightClickFunction(clickedActor)


    #######################################################################
    def _mousemiddle(self, iren, event):

        x, y = iren.GetEventPosition()

        renderer = iren.FindPokedRenderer(x, y)
        self.renderer = renderer

        picker = vtk.vtkPropPicker()
        picker.PickProp(x, y, renderer)
        clickedActor = picker.GetActor()

        # check if any button objects were created
        clickedActor2D = picker.GetActor2D()
        if clickedActor2D:
            for bt in self.buttons:
                if clickedActor2D == bt.actor:
                    bt.function()
                    break

        if not clickedActor:
            clickedActor = picker.GetAssembly()

        if not clickedActor:
            clickedActor = picker.GetProp3D()

        if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
            return
        self.clickedActor = clickedActor
        self.picked3d = picker.GetPickPosition()
        self.picked2d = np.array([x,y])

        if self.mouseMiddleClickFunction:
            self.mouseMiddleClickFunction(self.clickedActor)


    #######################################################################
    def _keypress(self, iren, event):

        #NB: qt creates and passes a vtkGenericRenderWindowInteractor

        key = iren.GetKeySym()
        #utils.vedo.printc('Pressed key:', self.keyheld, key, c='y', box='-')

        if key in ["Shift_L", "Control_L", "Super_L", "Alt_L",
                   "Shift_R", "Control_R", "Super_R", "Alt_R", "Menu"]:
            self.keyheld = key

        if key in ["q", "space", "Return"]:
            #iren.ExitCallback()
            return

        elif key == "Escape":
            vedo.printc('\nClosing window. Plotter.escaped is set to True.', c='r')
            self.escaped = True # window will be escaped ASAP
            iren.ExitCallback()
            return

        elif key == "F1":
            vedo.printc('\nExecution aborted. Exiting python kernel now.', c='r')
            iren.ExitCallback()
            sys.exit(0)

        # if ("Control_" in self.keyheld) and key=="c":
        #     print('ctrl-c')

        #############################################################
        ### now intercept custom observer ###########################
        #############################################################
        if self.keyPressFunction:
            if key not in ["Shift_L", "Control_L", "Super_L", "Alt_L",
                           "Shift_R", "Control_R", "Super_R", "Alt_R"]:
                self.keyPressFunction(key)
                return
        #############################################################

        if key == "Down":
            if self.clickedActor in self.getMeshes():
                self.clickedActor.GetProperty().SetOpacity(0.02)
                bfp = self.clickedActor.GetBackfaceProperty()
                if bfp and hasattr(self.clickedActor, "_bfprop"):
                    self.clickedActor._bfprop = bfp  # save it
                    self.clickedActor.SetBackfaceProperty(None)
            else:
                for a in self.getMeshes():
                    if a.GetPickable():
                        a.GetProperty().SetOpacity(0.02)
                        bfp = a.GetBackfaceProperty()
                        if bfp and hasattr(a, "_bfprop"):
                            a._bfprop = bfp
                            a.SetBackfaceProperty(None)

        elif key == "Left":
            if self.clickedActor in self.getMeshes():
                ap = self.clickedActor.GetProperty()
                aal = max([ap.GetOpacity() * 0.75, 0.01])
                ap.SetOpacity(aal)
                bfp = self.clickedActor.GetBackfaceProperty()
                if bfp and hasattr(self.clickedActor, "_bfprop"):
                    self.clickedActor._bfprop = bfp
                    self.clickedActor.SetBackfaceProperty(None)
            else:
                for a in self.getMeshes():
                    if a.GetPickable():
                        ap = a.GetProperty()
                        aal = max([ap.GetOpacity() * 0.75, 0.01])
                        ap.SetOpacity(aal)
                        bfp = a.GetBackfaceProperty()
                        if bfp and hasattr(a, "_bfprop"):
                            a._bfprop = bfp
                            a.SetBackfaceProperty(None)

        elif key == "Right":
            if self.clickedActor in self.getMeshes():
                ap = self.clickedActor.GetProperty()
                aal = min([ap.GetOpacity() * 1.25, 1.0])
                ap.SetOpacity(aal)
                if aal == 1 and hasattr(self.clickedActor, "_bfprop") \
                  and self.clickedActor._bfprop:
                    # put back
                    self.clickedActor.SetBackfaceProperty(self.clickedActor._bfprop)
            else:
                for a in self.getMeshes():
                    if a.GetPickable():
                        ap = a.GetProperty()
                        aal = min([ap.GetOpacity() * 1.25, 1.0])
                        ap.SetOpacity(aal)
                        if aal == 1 and hasattr(a, "_bfprop") and a._bfprop:
                            a.SetBackfaceProperty(a._bfprop)

        elif key == "slash" or key == "Up":
            if self.clickedActor in self.getMeshes():
                self.clickedActor.GetProperty().SetOpacity(1)
                if hasattr(self.clickedActor, "_bfprop") and self.clickedActor._bfprop:
                    self.clickedActor.SetBackfaceProperty(self.clickedActor._bfprop)
            else:
                for a in self.getMeshes():
                    if a.GetPickable():
                        a.GetProperty().SetOpacity(1)
                        if hasattr(a, "_bfprop") and a._bfprop:
                            a.clickedActor.SetBackfaceProperty(a._bfprop)

        elif key == "P":
            if self.clickedActor in self.getMeshes():
                acts = [self.clickedActor]
            else:
                acts = self.getMeshes()
            for ia in acts:
                if ia.GetPickable():
                    try:
                        ps = ia.GetProperty().GetPointSize()
                        if ps > 1:
                            ia.GetProperty().SetPointSize(ps - 1)
                        ia.GetProperty().SetRepresentationToPoints()
                    except AttributeError:
                        pass

        elif key == "p":
            if self.clickedActor in self.getMeshes():
                acts = [self.clickedActor]
            else:
                acts = self.getMeshes()
            for ia in acts:
                if ia.GetPickable():
                    try:
                        ps = ia.GetProperty().GetPointSize()
                        ia.GetProperty().SetPointSize(ps + 2)
                        ia.GetProperty().SetRepresentationToPoints()
                    except AttributeError:
                        pass

        elif key == "w":
            if self.clickedActor and self.clickedActor in self.getMeshes():
                self.clickedActor.GetProperty().SetRepresentationToWireframe()
            else:
                for a in self.getMeshes():
                    if a and a.GetPickable():
                        if a.GetProperty().GetRepresentation() == 1:  # toggle
                            a.GetProperty().SetRepresentationToSurface()
                        else:
                            a.GetProperty().SetRepresentationToWireframe()

        elif key == "r":
            self.renderer.ResetCamera()


        elif key == "h":
            vedo.docs.tips()
            return

        elif key == "a":
            iren.ExitCallback()
            cur = iren.GetInteractorStyle()
            if isinstance(cur, vtk.vtkInteractorStyleTrackballCamera):
                vedo.printc("\nInteractor style changed to TrackballActor")
                vedo.printc("  you can now move and rotate individual meshes:")
                vedo.printc("  press X twice to save the repositioned mesh,")
                vedo.printc("  press 'a' to go back to normal style.")
                iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
            else:
                iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            iren.Start()
            return

        elif key == "A": # toggle antialiasing
            msam = settings.plotter_instance.window.GetMultiSamples()
            if not msam:
                settings.plotter_instance.window.SetMultiSamples(8)
            else:
                settings.plotter_instance.window.SetMultiSamples(0)
            msam = settings.plotter_instance.window.GetMultiSamples()
            if msam:
                vedo.printc(f'Antialiasing is now set to {msam} samples', c=bool(msam))
            else:
                vedo.printc('Antialiasing is now disabled', c=bool(msam))

        elif key == "D": # toggle depthpeeling
            udp = not settings.plotter_instance.renderer.GetUseDepthPeeling()
            settings.plotter_instance.renderer.SetUseDepthPeeling(udp)
            #settings.plotter_instance.renderer.SetUseDepthPeelingForVolumes(udp)
            # print(settings.plotter_instance.window.GetAlphaBitPlanes())
            if udp:
                settings.plotter_instance.window.SetAlphaBitPlanes(1)
                settings.plotter_instance.renderer.SetMaximumNumberOfPeels(settings.maxNumberOfPeels)
                settings.plotter_instance.renderer.SetOcclusionRatio(settings.occlusionRatio)
            settings.plotter_instance.interactor.Render()
            wasUsed = settings.plotter_instance.renderer.GetLastRenderingUsedDepthPeeling()
            rnr = self.renderers.index(settings.plotter_instance.renderer)
            vedo.printc(f'Depth peeling is now set to {udp} for renderer nr.{rnr}', c=udp)
            if not wasUsed and udp:
                vedo.printc('\t...but last rendering did not actually used it!', c=udp, invert=True)
            return

        elif key == "j":
            iren.ExitCallback()
            cur = iren.GetInteractorStyle()
            if isinstance(cur, vtk.vtkInteractorStyleJoystickCamera):
                iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            else:
                vedo.printc("\nInteractor style changed to Joystick,", end="")
                vedo.printc(" press j to go back to normal.")
                iren.SetInteractorStyle(vtk.vtkInteractorStyleJoystickCamera())
            iren.Start()
            return

        elif key == "S":
            vedo.io.screenshot("screenshot.png")
            vedo.printc("\camera Saved rendering window as screenshot.png", c="blue")
            return

        elif key == "C":
            cam = self.renderer.GetActiveCamera()
            vedo.printc('\n###################################################', c='y')
            vedo.printc('## Template python code to position this camera: ##', c='y')
            vedo.printc('cam = dict(pos='          +utils.precision(cam.GetPosition(),4)+',', c='y')
            vedo.printc('           focalPoint='   +utils.precision(cam.GetFocalPoint(),4)+',', c='y')
            vedo.printc('           viewup='       +utils.precision(cam.GetViewUp(),4)+',', c='y')
            vedo.printc('           distance='     +utils.precision(cam.GetDistance(),4)+',', c='y')
            vedo.printc('           clippingRange='+utils.precision(cam.GetClippingRange(),4)+')', c='y')
            vedo.printc('show(mymeshes, camera=cam)', c='y')
            vedo.printc('\n### OR equivalently: ##############################', c='y')
            vedo.printc('plt = vedo.Plotter()\n...', c='y')
            vedo.printc('plt.camera.SetPosition(',   [round(e, 3) for e in cam.GetPosition()],  ')', c='y')
            vedo.printc('plt.camera.SetFocalPoint(', [round(e, 3) for e in cam.GetFocalPoint()], ')', c='y')
            vedo.printc('plt.camera.SetViewUp(',     [round(e, 3) for e in cam.GetViewUp()], ')', c='y')
            vedo.printc('plt.camera.SetDistance(',   round(cam.GetDistance(), 3), ')', c='y')
            vedo.printc('plt.camera.SetClippingRange(',
                                    [round(e, 3) for e in cam.GetClippingRange()], ')', c='y')
            vedo.printc('plt.show(mymeshes, resetcamera=False)', c='y')
            vedo.printc('###################################################', c='y')
            return

        elif key == "s":
            if self.clickedActor and self.clickedActor in self.getMeshes():
                self.clickedActor.GetProperty().SetRepresentationToSurface()
            else:
                for a in self.getMeshes():
                    if a and a.GetPickable():
                        a.GetProperty().SetRepresentationToSurface()

        elif key == "1":
            self._icol += 1
            if isinstance(self.clickedActor, vedo.Points):
                self.clickedActor.GetMapper().ScalarVisibilityOff()
                self.clickedActor.GetProperty().SetColor(vedo.colors.colors1[(self._icol) % 10])

        elif key == "2":
            self._icol += 1
            if isinstance(self.clickedActor, vedo.Points):
                self.clickedActor.GetMapper().ScalarVisibilityOff()
                self.clickedActor.GetProperty().SetColor(vedo.colors.colors2[(self._icol) % 10])

        elif key == "3":
            if isinstance(self.clickedActor, vedo.Mesh):
                if self.clickedActor._current_texture_name in settings.textures:
                    i = settings.textures.index(self.clickedActor._current_texture_name)
                    i = (i+1) % len(settings.textures)
                    self.clickedActor.texture(settings.textures[i])
                    self.clickedActor._current_texture_name = settings.textures[i]
                elif not self.clickedActor._current_texture_name:
                    self.clickedActor.texture(settings.textures[0])
                    self.clickedActor._current_texture_name = settings.textures[0]

        elif key == "4":
            if self.clickedActor:
                acts = [self.clickedActor]
            else:
                acts = self.getMeshes()
            for ia in acts:
                cmap_name = ia._cmap_name
                if not cmap_name:
                    cmap_name = "rainbow"
                if isinstance(ia, vedo.pointcloud.Points):
                    arnames = ia.pointdata.keys()
                    if len(arnames):
                        arnam =  arnames[ia._scals_idx]
                        if arnam and ("normals" not in arnam.lower()): # exclude normals
                            ia.cmap(cmap_name, arnam, on="points")
                            vedo.printc("..active point data set to:", arnam, c='g', bold=0)
                            ia._scals_idx += 1
                            if ia._scals_idx >= len(arnames):
                                ia._scals_idx = 0
                    else:
                        arnames = ia.celldata.keys()
                        if len(arnames):
                            arnam =  arnames[ia._scals_idx]
                            if arnam and ("normals" not in arnam.lower()): # exclude normals
                                ia.cmap(cmap_name, arnam, on="cells")
                                vedo.printc("..active cell array set to:", arnam, c='g', bold=0)
                                ia._scals_idx += 1
                                if ia._scals_idx >= len(arnames):
                                    ia._scals_idx = 0

        elif key == "5":
            bgc = np.array(self.renderer.GetBackground()).sum() / 3
            if bgc <= 0:
                bgc = 0.223
            elif 0 < bgc < 1:
                bgc = 1
            else:
                bgc = 0
            self.renderer.SetBackground(bgc, bgc, bgc)

        elif key == "6":
            bg2cols = ['lightyellow', 'darkseagreen', 'palegreen',
                       'steelblue','lightblue', 'cadetblue','lavender',
                       'white', 'blackboard', 'black']
            bg2name = vedo.getColorName(self.renderer.GetBackground2())
            if bg2name in bg2cols:
                idx = bg2cols.index(bg2name)
            else:
                idx = 4
            if idx is not None:
                bg2name_next = bg2cols[(idx+1)%(len(bg2cols)-1)]
            if not bg2name_next:
                self.renderer.GradientBackgroundOff()
            else:
                self.renderer.GradientBackgroundOn()
                self.renderer.SetBackground2(vedo.getColor(bg2name_next))

        elif key in ["plus", "equal", "KP_Add", "minus", "KP_Subtract"]:  # cycle axes style
            clickedr = self.renderers.index(self.renderer)
            if self.axes_instances[clickedr]:
                if hasattr(self.axes_instances[clickedr], "EnabledOff"):  # widget
                    self.axes_instances[clickedr].EnabledOff()
                else:
                    try:
                        self.renderer.RemoveActor(self.axes_instances[clickedr])
                    except:
                        pass
                self.axes_instances[clickedr] = None
            if not self.axes: self.axes=0
            if isinstance(self.axes, dict):
                self.axes=1
            if key in ["minus", "KP_Subtract"]:
                if settings.useParallelProjection == False and self.axes==0:
                    self.axes -= 1 # jump ruler doesnt make sense in perspective mode
                addons.addGlobalAxes(axtype=(self.axes-1)%14, c=None)
            else:
                if settings.useParallelProjection == False and self.axes==12:
                    self.axes += 1 # jump ruler doesnt make sense in perspective mode
                addons.addGlobalAxes(axtype=(self.axes+1)%14, c=None)
            self.interactor.Render()

        elif "KP_" in key:  # change axes style
            asso = {
                    "KP_Insert":0, "KP_0":0,
                    "KP_End":1,    "KP_1":1,
                    "KP_Down":2,   "KP_2":2,
                    "KP_Next":3,   "KP_3":3,
                    "KP_Left":4,   "KP_4":4,
                    "KP_Begin":5,  "KP_5":5,
                    "KP_Right":6,  "KP_6":6,
                    "KP_Home":7,   "KP_7":7,
                    "KP_Up":8,     "KP_8":8,
                    "KP_Prior":9,  "KP_9":9,
                    }
            clickedr = self.renderers.index(self.renderer)
            if key in asso.keys():
                if self.axes_instances[clickedr]:
                    if hasattr(self.axes_instances[clickedr], "EnabledOff"):  # widget
                        self.axes_instances[clickedr].EnabledOff()
                    else:
                        try:
                            self.renderer.RemoveActor(self.axes_instances[clickedr])
                        except:
                            pass
                    self.axes_instances[clickedr] = None
                addons.addGlobalAxes(axtype=asso[key], c=None)
                self.interactor.Render()

        if key == "O":
            self.renderer.RemoveLight(self._extralight)
            self._extralight = None

        elif key == "o":
            vbb, sizes, _, _ = addons.computeVisibleBounds()
            cm = utils.vector((vbb[0]+vbb[1])/2, (vbb[2]+vbb[3])/2, (vbb[4]+vbb[5])/2)
            if not self._extralight:
                vup = self.renderer.GetActiveCamera().GetViewUp()
                pos = cm + utils.vector(vup)*utils.mag(sizes)
                self._extralight = addons.Light(pos, focalPoint=cm)
                self.renderer.AddLight(self._extralight)
                print("Press again o to rotate light source, or O to remove it.")
            else:
                cpos = utils.vector(self._extralight.GetPosition())
                x, y, z = self._extralight.GetPosition() - cm
                r,th,ph = utils.cart2spher(x,y,z)
                th += 0.2
                if th>np.pi: th=np.random.random()*np.pi/2
                ph += 0.3
                cpos = utils.spher2cart(r, th,ph) + cm
                self._extralight.SetPosition(cpos)

            self.window.Render()

        elif key == "l":
            if self.clickedActor in self.getMeshes():
                acts = [self.clickedActor]
            else:
                acts = self.getMeshes()
            for ia in acts:
                if not ia.GetPickable():
                    continue
                try:
                    ev = ia.GetProperty().GetEdgeVisibility()
                    ia.GetProperty().SetEdgeVisibility(not ev)
                    ia.GetProperty().SetRepresentationToSurface()
                    ia.GetProperty().SetLineWidth(0.1)
                except AttributeError:
                    pass

        elif key == "k": # lightings
            if self.clickedActor in self.getMeshes():
                acts = [self.clickedActor]
            else:
                acts = self.getMeshes()
            shds = ('default',
                    'metallic',
                    'plastic',
                    'shiny',
                    'glossy',
                    'off')
            for ia in acts:
                if ia.GetPickable():
                    try:
                        lnr = (ia._ligthingnr+1)%6
                        ia.lighting(shds[lnr])
                        ia._ligthingnr = lnr
                        # vedo.printc('-> lighting set to:', shds[lnr], c='g', bold=0)
                    except AttributeError:
                        pass

        elif key == "K": # shading
            if self.clickedActor in self.getMeshes():
                acts = [self.clickedActor]
            else:
                acts = self.getMeshes()
            for ia in acts:
                if ia.GetPickable() and isinstance(ia, vedo.Mesh):
                    ia.computeNormals(cells=False)
                    intrp = ia.GetProperty().GetInterpolation()
                    # print(intrp, ia.GetProperty().GetInterpolationAsString())
                    if intrp > 0:
                        ia.GetProperty().SetInterpolation(0) #flat
                    else:
                        ia.GetProperty().SetInterpolation(2) #phong

        elif key == "n":  # show normals to an actor
            if self.clickedActor in self.getMeshes():
                if self.clickedActor.GetPickable():
                    self.renderer.AddActor(vedo.shapes.NormalLines(self.clickedActor))
                    iren.Render()
            else:
                print("Click an actor and press n to add normals.")


        elif key == "x":
            if self.justremoved is None:
                if self.clickedActor in self.getMeshes() \
                  or isinstance(self.clickedActor, vtk.vtkAssembly):
                    self.justremoved = self.clickedActor
                    self.renderer.RemoveActor(self.clickedActor)
            else:
                self.renderer.AddActor(self.justremoved)
                self.renderer.Render()
                self.justremoved = None

        elif key == "X":
            if self.clickedActor:
                if not self.cutterWidget:
                    addons.addCutterTool(self.clickedActor)
                else:
                    if isinstance(self.clickedActor, vtk.vtkActor):
                        fname = "clipped.vtk"
                        w = vtk.vtkPolyDataWriter()
                        w.SetInputData(self.clickedActor.polydata())
                        w.SetFileName(fname)
                        w.Write()
                        vedo.printc("\save Saved file:", fname, c="m")
                        self.cutterWidget.Off()
                        self.cutterWidget = None
            else:
                for a in self.actors:
                    if isinstance(a, vtk.vtkVolume):
                        addons.addCutterTool(a)
                        return

                vedo.printc("Click object and press X to open the cutter box widget.", c=4)

        elif key == "E":
            vedo.printc("\camera Exporting 3D window to file", c="blue", end="")
            vedo.io.exportWindow('scene.npz')
            vedo.printc(". Try:\n> vedo scene.npz", c="blue")
            settings.plotter_instance.interactor.Start()

        elif key == "F12":
            vedo.io.exportWindow('scene.x3d')
            vedo.printc("Try: firefox scene.html", c="blue")

        elif key == "i":  # print info
            if self.clickedActor:
                utils.printInfo(self.clickedActor)
            else:
                utils.printInfo(self)

        elif key == "I":  # print color under the mouse
            x, y = iren.GetEventPosition()
            rgb = vedo.colors.colorPicker([x,y], self)
            if rgb is None: return
            vedo.printc('Pixel', [x,y], 'has RGB[',  end='')
            vedo.printc('█', c=[rgb[0],0,0], end='')
            vedo.printc('█', c=[0,rgb[1],0], end='')
            vedo.printc('█', c=[0,0,rgb[2]], end='')
            vedo.printc('] = ', end='')
            cnm = vedo.getColorName(rgb)
            if np.sum(rgb) < 150:
                vedo.printc(rgb.tolist(), vedo.colors.rgb2hex(np.array(rgb)/255), c='w',
                       bc=rgb, invert=1, end='')
                vedo.printc('  ~ '+cnm, invert=1, c='w')
            else:
                vedo.printc(rgb.tolist(), vedo.colors.rgb2hex(np.array(rgb)/255), c=rgb, end='')
                vedo.printc('  ~ '+cnm, c=cnm)

        if iren:
            iren.Render()
        return

    ######################################
    def _keyrelease(self, iren, event):
        key = iren.GetKeySym()
        # print(iren.GetShiftKey())
        # utils.vedo.printc('Released key:', key, c='v', box='-')
        if key in ["Shift_L", "Control_L", "Super_L", "Alt_L",
                   "Shift_R", "Control_R", "Super_R", "Alt_R", "Menu"]:
            self.keyheld = ''
        return

