#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import sys
import time

import numpy as np
import vedo
import vedo.addons as addons
import vedo.backends as backends
import vedo.utils as utils
from vedo import settings
import vtk

__doc__ = """
This module defines the main class Plotter to manage actors and 3D rendering
.. image:: https://vedo.embl.es/images/basic/multirenderers.png
"""

__all__ = [
        "Plotter",
        "show",
        "clear",
        "interactive",
        "close",
]


##################################################################################
def _embedWindow(backend='ipyvtk'):
    # check availability of backend by just returning its name

    if not backend:
        return None ####################

    else:

        if any(['SPYDER' in name for name in os.environ]):
            return None

        try:
            get_ipython()
        except NameError:
            return None

    backend = backend.lower()

    if backend=='k3d':
        try:
            import k3d
            return backend
            if k3d._version.version_info != (2, 7, 4):
                vedo.logger.warning('Only k3d version 2.7.4 is currently supported')

        except ModuleNotFoundError:
            vedo.logger.error('Could not load k3d try:\n> pip install k3d==2.7.4')
            print(flush=True)

    elif 'ipygany' in backend: # ipygany
        try:
            import ipygany
            return backend
        except ModuleNotFoundError:
            vedo.logger.error('Could not load ipygany try:\n> pip install ipygany')
            print(flush=True)

    elif 'itk' in backend: # itkwidgets
        try:
            import itkwidgets
            return backend
        except ModuleNotFoundError:
            vedo.logger.error('Could not load itkwidgets try:\n> pip install itkwidgets')
            print(flush=True)

    elif backend.lower() == '2d':
        return backend

    elif backend =='panel':
        try:
            import panel
            panel.extension('vtk')
            return backend
        except:
            vedo.logger.error('Could not load panel try:\n> pip install panel')

    elif 'ipyvtk' in backend:
        try:
            from ipyvtklink.viewer import ViewInteractiveWidget
            return backend
        except ModuleNotFoundError:
            vedo.logger.error('Could not load ipyvtklink try:\n> pip install ipyvtklink')
            print(flush=True)

    else:
        vedo.logger.error("Unknown backend" + str(backend))
        raise RuntimeError()

    return None


########################################################################################################
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
        mode=0,
        q=False,
        new=False,
        backend=None,
    ):
    """
    Create on the fly an instance of class Plotter and show the object(s) provided.

    Allowed input objects types are:
        ``str, Mesh, Volume, Picture, Assembly
        vtkPolyData, vtkActor, vtkActor2D, vtkImageActor,
        vtkAssembly or vtkVolume``

    Parameters
    ----------

    at : int, optional
        number of the renderer to plot to, in case of more than one exists

    shape : list, str, optional
        Number of sub-render windows inside of the main window. E.g.:
        specify two across with shape=(2,1) and a two by two grid
        with shape=(2, 2).  By default there is only one renderer.

        Can also accept a shape as string descriptor. E.g.:

        - shape="3|1" means 3 plots on the left and 1 on the right,
        - shape="4/2" means 4 plots on top of 2 at bottom.

    axes : int, optional
        set the type of axes to be shown:

        - 0,  no axes
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

        Axis type-1 can be fully customized by passing a dictionary.
        Check addons.Axes() for the full list of options.

    azimuth/elevation/roll : float, optional
        move camera accordingly the specified value

    viewup: str, list
        either ['x', 'y', 'z'] or a vector to set vertical direction

    resetcam : bool
        re-adjust camera position to fit objects

    camera : dict, vtkCamera
        camera parameters can further be specified with a dictionary
        assigned to the ``camera`` keyword (E.g. `show(camera={'pos':(1,2,3), 'thickness':1000,})`)

        - pos, `(list)`,  the position of the camera in world coordinates

        - focalPoint `(list)`, the focal point of the camera in world coordinates

        - viewup `(list)`, the view up direction for the camera

        - distance `(float)`, set the focal point to the specified distance from the camera position.

        - clippingRange `(float)`, distance of the near and far clipping planes along the direction of projection.

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

    interactive : bool
        pause and interact with window (True) or continue execution (False)

    rate : float
        maximum rate of `show()` in Hertz

    mode : int, str
        set the type of interaction

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

    q : bool
        force program to quit after `show()` command returns.

    new : bool
        if set to `True`, a call to show will instantiate
        a new Plotter object (a new window) instead of reusing the first created.
    """
    if len(actors) == 0:
        actors = None
    elif len(actors) == 1:
        actors = actors[0]
    else:
        actors = utils.flatten(actors)

    if vedo.plotter_instance and not new: # Plotter exists
        plt = vedo.plotter_instance

    else:                                 # Plotter must be created

        if utils.isSequence(at):          # user passed a sequence for "at"

            if not utils.isSequence(actors):
                vedo.logger.error("in show() input must be a list.")
                raise RuntimeError()
            if len(at) != len(actors):
                vedo.logger.error("in show() lists 'input' and 'at' must have equal lengths")
                raise RuntimeError()
            if shape==(1, 1) and N is None:
                N = max(at) + 1

        elif at is None and (N or shape != (1, 1)):

            if not utils.isSequence(actors):
                e = "in show(), N or shape is set, but input is not a sequence\n"
                e+= "              you may need to specify e.g. at=0"
                vedo.logger.error(e)
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
            backend=backend,
        )

    # use _plt_to_return because plt.show() can return a k3d/panel plot
    _plt_to_return = None

    if utils.isSequence(at):
        for i, act in enumerate(actors):
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
                    bg=bg,
                    bg2=bg2,
                    axes=axes,
                    q=q,
            )

        if interactive or len(at)==N \
            or (isinstance(shape[0],int) and len(at)==shape[0]*shape[1]):
            # note that shape can be a string
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
                    mode=mode,
                    bg=bg,
                    bg2=bg2,
                    axes=axes,
                    q=q,
        )

    return _plt_to_return


def interactive():
    """Start the rendering window interaction mode."""
    if vedo.plotter_instance:
        if vedo.plotter_instance.escaped: # just return
            return vedo.plotter_instance
        if hasattr(vedo.plotter_instance, 'interactor'):
            if vedo.plotter_instance.interactor:
                vedo.plotter_instance.interactor.Start()
    return vedo.plotter_instance

def clear(actor=None, at=None):
    """
    Clear specific actor or list of actors from the current rendering window.
    Keyword ``at`` specify the reneder to be cleared.
    """
    if not vedo.plotter_instance:
        return
    vedo.plotter_instance.clear(actor, at)
    return vedo.plotter_instance

def close():
    """Close the last created Plotter instance if it exists."""
    if not vedo.plotter_instance:
        return
    vedo.plotter_instance.close()
    return


########################################################################
class Plotter:
    """
    Main class to manage actors.

    Parameters
    ----------

    shape : str, list
        shape of the grid of renderers in format (rows, columns). Ignored if N is specified.

    N : int
        number of desired renderers arranged in a grid automatically.

    pos : list
        (x,y) position in pixels of top-left corner of the rendering window on the screen

    size : str, list
        size of the rendering window. If 'auto', guess it based on screensize.

    screensize : list
        physical size of the monitor screen in pixels

    bg : color, str
        background color or specify jpg image file name with path

    bg2 : color
        background color of a gradient towards the top

    axes : int
          - 0,  no axes
          - 1,  draw three gray grid walls
          - 2,  show cartesian axes from (0,0,0)
          - 3,  show positive range of cartesian axes from (0,0,0)
          - 4,  show a triad at bottom left
          - 5,  show a cube at bottom left
          - 6,  mark the corners of the bounding box
          - 7,  draw a 3D ruler at each side of the cartesian axes
          - 8,  show the VTK CubeAxesActor object
          - 9,  show the bounding box outLine,
          - 10, show three circles representing the maximum bounding box,
          - 11, show a large grid on the x-y plane (use with zoom=8)
          - 12, show polar axes.
          - 13, draw a simple ruler at the bottom of the window

    Note that Axes type-1 can be fully customized by passing a dictionary axes=dict().
    Check Axes() for the available options.

    sharecam : bool
        if False each renderer will have an independent vtkCamera

    interactive : bool
        if True will stop after show() to allow interaction w/ window

    offscreen : bool
        if True will not show the rendering window

    qtWidget : QVTKRenderWindowInteractor
        render in a Qt-Widget using an QVTKRenderWindowInteractor.
        Overrides offscreen to True.
        Overrides interactive to False.
        See Also: Example qt_windows1.py and qt_windows2.py
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
            backend=None,
        ):

        vedo.notebookBackend = _embedWindow(backend)

        vedo.plotter_instance = self

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
        self._interactive = interactive  # allows to interact with renderer
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
        self.camera = None
        self.keyheld = ''

        #####################################################################
        notebookBackend = vedo.notebookBackend
        if notebookBackend:
            if notebookBackend == '2d':
                self.offscreen = True
                if self.size == "auto":
                    self.size = (900, 700)

            elif notebookBackend == "k3d" or "ipygany" in notebookBackend:
                self._interactive = False
                self.interactor = None
                self.window = None
                self.camera = None # let the backend choose
                if self.size == "auto":
                    self.size = (1000, 1000)
                #############################################################
                return ######################################################
                #############################################################
        #####################################################################

        # build the rendering window:
        self.window = vtk.vtkRenderWindow()
        self.escaped = False

        self.window.GlobalWarningDisplayOff()
        self.window.SetWindowName(self.title)

        self._repeating_timer_id = None
        self._timer_event_id = None

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

            ### BUG in GetScreenSize in VTK 9.1.0
            ### https://discourse.vtk.org/t/vtk9-1-0-problems/7094/3
            if settings.hackCallScreenSize: # True

                vtkvers = vedo.vtk_version
                if not self.offscreen and (vtkvers[0]<9 or vtkvers[0]==9 and vtkvers[1]==0):
                     aus = self.window.GetScreenSize()
                     if aus and len(aus) == 2 and aus[0] > 100 and aus[1] > 100:  # seems ok
                         if aus[0] / aus[1] > 2:  # looks like there are 2 or more screens
                             screensize = (int(aus[0] / 2), aus[1])
                         else:
                             screensize = aus

        x, y = screensize

        if N:  # N = number of renderers. Find out the best

            if shape != (1, 1):  # arrangement based on minimum nr. of empty renderers
                vedo.logger.warning("having set N, shape is ignored.")

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
            self.camera = self.renderer.GetActiveCamera()

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
            self._interactive = False
            self.interactor = None
            ########################
            return #################
            ########################

        if vedo.notebookBackend == "panel":
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
            self._repeatingtimer_id = self.interactor.CreateRepeatingTimer(1)
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

    def __enter__(self):
        # context manager like in "with Plotter() as plt:"
        return self

    def __exit__(self, *args, **kwargs):
        # context manager like in "with Plotter() as plt:"
        self.close()
        return None


    def load(self, filename, unpack=True, force=False):
        """
        Load objects from file.
        The output will depend on the file extension. See examples below.

        Parameters
        ----------
        unpack : bool
            only for multiblock data, if True returns a flat list of objects.

        force : bool
            when downloading a file ignore any previous cached downloads and force a new one.

        Example:
            .. code-block:: python

                from vedo import *
                # Return a list of 2 Mesh
                meshes = load([dataurl+'250.vtk', dataurl+'290.vtk'])
                show(meshes)
                # Return a list of meshes by reading all files in a directory
                # (if directory contains DICOM files then a Volume is returned)
                meshes = load('mydicomdir/')
                show(meshes)
                # Return a Volume
                vol = load(dataurl+'embryo.slc')
                vol.show()
        """
        acts = vedo.io.load(filename, unpack, force)
        if utils.isSequence(acts):
            self.actors += acts
        else:
            self.actors.append(acts)
        return acts

    def at(self, nren):
        """Select the current renderer number."""
        self.renderer = self.renderers[nren]
        return self

    def add(self, *actors, at=None, render=True, resetcam=False):
        """
        Append the input objects to the internal list of actors to be shown.
        This method is typically used in loops or callback functions.

        Parameters
        ----------
        at : int
            add the object at the specified renderer

        render : bool
            render the scene after adding the objects (True by default)
        """
        if at is not None:
            ren = self.renderers[at]
        else:
            ren = self.renderer

        actors = utils.flatten(actors)
        actors = self._scan_input(actors)

        for a in actors:
            if a not in self.actors:
                self.actors.append(a)
            if ren:
                ren.AddActor(a)
        if render:
            self.render(resetcam=resetcam)
        return self


    def remove(self, *actors, at=None, render=False, resetcam=False):
        """
        Remove input object to the internal list of actors to be shown.
        This method is typically used in loops or callback functions.

        Parameters
        ----------
        at : int
            remove the object at the specified renderer

        render : bool
            render the scene after removing the objects (False by default).
        """
        if at is not None:
            ren = self.renderers[at]
        else:
            ren = self.renderer

        actors = utils.flatten(actors)
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

    def pop(self, at=None):
        """
        Remove the last added object from the rendering window.
        This method is typically used in loops or callback functions.
        """
        if len(self.actors):
            self.remove(self.actors[-1], at)
        return self

    def render(self, resetcam=False):
        """Render the scene. This method is typically used in loops or callback functions."""
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

        # if at is not None: # disable all except i==at
        #     self.window.EraseOff()
        #     if at < 0:
        #         at = at + len(self.renderers) +1
        #     for i, ren in enumerate(self.renderers):
        #         if i != at:
        #             ren.DrawOff()

        if vedo.vtk_version[0] == 9 and "Darwin" in vedo.sys_platform:
            for a in self.actors:
                if isinstance(a, vtk.vtkVolume):
                    self.window.SetMultiSamples(0) # to fix mac OSX BUG vtk9
                    break

        self.camera = self.renderer.GetActiveCamera()
        if resetcam:
            self.renderer.ResetCamera()

        if settings.allowInteraction:
            self.allowInteraction()

        self.window.Render()

        # if at is not None: # re-enable all that were disabled
        #     for i, ren in enumerate(self.renderers):
        #         if i != at:
        #             ren.DrawOn()
        #     self.window.EraseOn()

        return self

    def interactive(self):
        """
        Start window interaction.
        Analogous to show(..., interactive=True).
        """
        if self.interactor and not self.escaped:
            self.interactor.Start()
        return self

    def enableErase(self, value=True):
        """Enable erasing the redering window between render() calls."""
        self.window.SetErase(value)
        return self

    def enableRenderer(self, at=None, value=True):
        """Enable a render() call to refresh this renderer."""
        if at is None:
            ren = self.renderer
        else:
            ren = self.renderers[at]
        ren.SetDraw(value)
        return self

    def useDepthPeeling(self, at=None, value=True):
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

    def background(self, c1=None, c2=None, at=None):
        """Set the color of the background for the current renderer.
        A different renderer index can be specified by keyword ``at``.

        Parameters
        ----------
        c1 : list, optional
            background main color.

        c2 : list, optional
            background color for the upper part of the window.

        at : int, optional
            renderer index.
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


    ####################################################
    def getMeshes(self, at=None, includeNonPickables=False):
        """
        Return a list of Meshes from the specified renderer.

        Parameters
        ----------
        at : int
            specify which renderer to look at.

        includeNonPickables : bool
            include non-pickable objects
        """
        if at is None:
            renderer = self.renderer
            at = self.renderers.index(renderer)
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

        Parameters
        ----------
        at : int
            specify which renderer to look at

        includeNonPickables : bool
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
        for i in range(acs.GetNumberOfItems()):
            a = acs.GetNextItem()
            if includeNonPickables or a.GetPickable():
                vols.append(a)
        return vols


    def resetCamera(self, tight=None):
        """
        Reset the camera position and zooming.
        If tight is specified the zooming reserves a padding space in the xy-plane
        expressed in percent of the average size.
        """
        if tight is None:
            self.renderer.ResetCamera()
        else:
            x0, x1, y0, y1, z0, z1 = self.renderer.ComputeVisiblePropBounds()

            cam = self.renderer.GetActiveCamera()
            cam.SetParallelProjection(True)

            self.renderer.ComputeAspect()
            aspect = self.renderer.GetAspect()
            angle = np.pi*cam.GetViewAngle()/180.
            dx, dy = (x1-x0)*0.999, (y1-y0)*0.999
            dist = max(dx/aspect[0], dy) / np.sin(angle/2) / 2

            cam.SetViewUp(0, 1, 0)
            cam.SetPosition(x0 + dx/2, y0 + dy/2, dist*(1+tight))
            cam.SetFocalPoint(x0 + dx/2, y0 + dy/2, 0)
            ps = max(dx/aspect[0], dy) / 2
            cam.SetParallelScale(ps*(1+tight))
            self.renderer.ResetCameraClippingRange(x0, x1, y0, y1, z0, z1)
        return self


    def moveCamera(self, camstart, camstop, fraction):
        """
        Takes as input two vtkCamera objects and set camera at an intermediate position:

        fraction=0 -> camstart,  fraction=1 -> camstop.

        camstart and camstop can also be dictionaries of format:

            dict(pos=..., focalPoint=..., viewup=..., distance=..., clippingRange=...)

        Press shift-C key in interactive mode to dump a python snipplet
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
            Renderer number.

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

        .. hint:: examples/basic/record_play.py
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

        .. hint:: examples/basic/record_play.py
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


    def parallelProjection(self, value=True, at=None):
        """
        Use parallel projection ``at`` a specified renderer.
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


    ##################################################################
    def addSlider2D(self,
                    sliderfunc,
                    xmin, xmax,
                    value=None,
                    pos=4,
                    title="",
                    font="",
                    titleSize=1,
                    c=None,
                    showValue=True,
                    delayed=False,
                    **options,
        ):
        """
        Add a slider widget which can call an external custom function.

        Parameters
        ----------
        sliderfunc :
            external function to be called by the widget

        xmin : float
            lower value of the slider range

        xmax :  float
            upper value of the slider range

        value : float
            current value of the slider range

        pos : list
            position corner number, horizontal [1-5] or vertical [11-15]
            it can also be specified by corners coordinates [(x1,y1), (x2,y2)]

        title : str
            title text

        titleSize : float
            title text scale [1.0]

        font : str
            title font

        showValue : bool
            if true current value is shown

        delayed : bool
            if True the callback is delayed to when the mouse is released

        alpha : float
            opacity of the scalar bar texts

        sliderLength : float
            slider length

        sliderWidth : float
            slider width

        endCapLength : float
            length of the end cap

        endCapWidth : float
            width of the end cap

        tubeWidth : float
            width of the tube

        titleHeight : float
            width of the title

        tformat : str
            format of the title

        .. hint:: [sliders1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders1.py), sliders2.py
            ..image:: https://user-images.githubusercontent.com/32848391/50738848-be033480-11d8-11e9-9b1a-c13105423a79.jpg
        """
        return addons.addSlider2D(sliderfunc, xmin, xmax, value,
                                  pos, title, font, titleSize, c, showValue, delayed, **options)

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

        Parameters
        ----------
        sliderfunc : function
            external function to be called by the widget

        pos1 : list
            first position 3D coordinates

        pos2 : list
            second position coordinates

        xmin : float
            lower value

        xmax : float
            upper value

        value : float
            initial value

        s : float
            label scaling factor

        t : float
            tube scaling factor

        title : str
            title text

        c : color
            slider color

        rotation : float
            title rotation around slider axis

        showValue : bool
            if True current value is shown

        .. hint:: [sliders3d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders3d.py)
            ..image:: https://user-images.githubusercontent.com/32848391/52859555-4efcf200-312d-11e9-9290-6988c8295163.png
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
        """
        Add a button to the renderer window.

        Parameters
        ----------
        states : list
            a list of possible states, e.g. ['On', 'Off']

        c : list
            a list of colors for each state

        bc : list
            a list of background colors for each state

        pos : list
            2D position in pixels from left-bottom corner

        size : float
            size of button font

        font : str
            font type

        bold : bool
            bold font face (False)

        italic :
            italic font face (False)

        alpha : float
            opacity level

        angle : float
            anticlockwise rotation in degrees

        .. hint:: [buttons.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/buttons.py)
            .. image:: https://user-images.githubusercontent.com/32848391/50738870-c0fe2500-11d8-11e9-9b78-92754f5c5968.jpg
        """
        return addons.addButton(fnc, states, c, bc, pos, size, font,
                                bold, italic, alpha, angle)

    def addSplineTool(self, points, pc='k', ps=8, lc='r4', ac='g5', lw=2, closed=False, interactive=True):
        """
        Add a spline tool to the current plotter. Nodes of the spline can be dragged in space
        with the mouse.
        Clicking on the line itself adds an extra point.
        Selecting a point and pressing del removes it.

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

        .. hint:: examples/basic/spline_tool.py
        """
        sw = addons.SplineTool(points, pc, ps, lc, ac, lw, closed)
        if self.interactor:
            sw.SetInteractor(self.interactor)
        else:
            vedo.logger.error("in addSplineTool(), No interactor found.")
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


    def addCutterTool(self, obj=None, mode='box', invert=False):
        """Create an interactive tool to cut away parts of a mesh or volume.

        Parameters
        ----------
        mode : str
            either "box", "plane" or "sphere"

        invert : bool
            invert selection (inside-out)

        .. hint:: [cutter.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/cutter.py)
            .. image:: https://user-images.githubusercontent.com/32848391/50738866-c0658e80-11d8-11e9-955b-551d4d8b0db5.jpg
        """
        return addons.addCutterTool(obj, mode, invert)

    def addIcon(self, icon, pos=3, size=0.08):
        """Add an inset icon mesh into the same renderer.

        Parameters
        ----------
        pos : int, list
            icon position in the range [1-4] indicating one of the 4 corners,
            or it can be a tuple (x,y) as a fraction of the renderer size.

        size : float
            size of the square inset.

        .. hint:: examples/other/icon.py
        """
        return addons.addIcon(icon, pos, size)

    def addGlobalAxes(self, axtype=None, c=None):
        """Draw axes on scene. Available axes types:

        Parameters
        ----------
        axtype : int
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
            .. code-block:: python

                from vedo import Box, show
                b = Box(pos=(0,0,0), length=80, width=90, height=70).alpha(0)

                show(b, axes={ 'xtitle':'Some long variable [a.u.]',
                               'numberOfDivisions':4,
                               # ...
                             }
                )

        .. hint::
            [customAxes1.py](https://github.com/marcomusy/vedo/blob/master/examples/pyplot/customAxes1.py)
            customAxes2.py, customAxes3.py, customIndividualAxes.py
            .. image:: https://user-images.githubusercontent.com/32848391/72752870-ab7d5280-3bc3-11ea-8911-9ace00211e23.png
        """
        addons.addGlobalAxes(axtype, c)
        return self

    def addLegendBox(self, **kwargs):
        """Add a legend to the top right.

        .. hint:: examples/basic/legendbox.py, examples/other/flag_labels.py
        """
        acts = self.getMeshes()
        lb = addons.LegendBox(acts, **kwargs)
        self.add(lb)
        return self

    def addShadows(self):
        """Add shadows at the current renderer."""
        shadows = vtk.vtkShadowMapPass()
        seq = vtk.vtkSequencePass()
        passes = vtk.vtkRenderPassCollection()
        passes.AddItem(shadows.GetShadowMapBakerPass())
        passes.AddItem(shadows)
        seq.SetPasses(passes)
        camerapass = vtk.vtkCameraPass()
        camerapass.SetDelegatePass(seq)
        self.renderer.SetPass(camerapass)
        return self

    def _addSkybox(self, hdrfile):
        # many hdr files are at https://polyhaven.com/all

        if utils.vtkVersionIsAtLeast(9):
            reader = vtk.vtkHDRReader()
            # Check the image can be read.
            if not reader.CanReadFile(hdrfile):
                vedo.logger.error(f"Cannot read HDR file {hdrfile}")
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
            vedo.logger.error("addSkyBox not supported in this VTK version. Skip.")

        return self

    def addRendererFrame(self, c=None, alpha=None, lw=None, padding=None):
        """
        Add a frame to the renderer subwindow

        Parameters
        ----------
        c : color
            color name or index

        alpha : float
            opacity level

        lw : int
            line width in pixels.

        padding : float
            padding space in pixels.
        """
        self.frames = addons.addRendererFrame(self, c, alpha,lw, padding)
        return self


    def addHoverLegend(self,
                       at=None,
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

        The created text object are stored in plotter.hoverLegends

        Parameters
        ----------
        c : color
            Text color. If None then black or white is chosen automatically

        pos : str
            text positioning

        font : str
            text font type

        s : float
            text size scale

        bg : color
            background color of the 2D box containing the text

        alpha : float
            box transparency

        precision : int
            number of significant digits

        maxlength : int
            maximum number of characters per line

        useInfo : bool
            visualize the content of the obj.info attribute

        .. hint:: examples/basic/hoverLegend.py, examples/pyplot/earthquake_browser.py
        """
        hoverLegend = vedo.shapes.Text2D('', pos=pos, font=font, c=c, s=s, alpha=alpha, bg=bg)

        if at is None:
            at = self.renderers.index(self.renderer)

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
                    pcol = self.colorPicker(evt.picked2d)
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

    #####################################################################
    def addScaleIndicator(self, pos=(0.7,0.05), s=0.02, length=2,
                          lw=4, c='k1', alpha=1, units='', gap=0.05):
        """
        Add a Scale Indicator. Only works in parallel mode (no perspective).

        Parameters
        ----------
        pos : list, optional
            fractional (x,y) position on the screen.

        s : float, optional
            size of the text.

        length : float, optional
            length of the line.

        units : str, optional
            string to show units.

        gap : float, optional
            separation of line and text.

        Example
        -------
            .. code-block:: python

                from vedo import settings, Cube, Plotter
                settings.useParallelProjection = True # or else it doesnt make sense!
                cube = Cube().alpha(0.2)
                plt = Plotter(size=(900,600), axes=dict(xtitle='x (um)'))
                plt.addScaleIndicator(units='um', c='blue4')
                plt.show(cube, "Scale indicator with units")
        """
        ppoints = vtk.vtkPoints()  # Generate the polyline
        psqr = [[0.0,gap], [length/10,gap]]
        dd = psqr[1][0] - psqr[0][0]
        for i, pt in enumerate(psqr):
                ppoints.InsertPoint(i, pt[0], pt[1], 0)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(psqr))
        for i in range(len(psqr)):
            lines.InsertCellPoint(i)
        pd = vtk.vtkPolyData()
        pd.SetPoints(ppoints)
        pd.SetLines(lines)

        wsx, wsy = self.window.GetSize()
        if not settings.useParallelProjection:
            vedo.logger.warning("addScaleIndicator called with useParallelProjection OFF. Skip.")
            return None

        rlabel = vtk.vtkVectorText()
        rlabel.SetText('scale')
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(rlabel.GetOutputPort())
        t = vtk.vtkTransform()
        t.Scale(s*wsy/wsx, s, 1)
        tf.SetTransform(t)

        app = vtk.vtkAppendPolyData()
        app.AddInputConnection(tf.GetOutputPort())
        app.AddInputData(pd)

        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputConnection(app.GetOutputPort())
        cs = vtk.vtkCoordinate()
        cs.SetCoordinateSystem(1)
        mapper.SetTransformCoordinate(cs)

        fractor = vtk.vtkActor2D()
        csys = fractor.GetPositionCoordinate()
        csys.SetCoordinateSystem(3)
        fractor.SetPosition(pos)
        fractor.SetMapper(mapper)
        fractor.GetProperty().SetColor(vedo.getColor(c))
        fractor.GetProperty().SetOpacity(alpha)
        fractor.GetProperty().SetLineWidth(lw)
        fractor.GetProperty().SetDisplayLocationToForeground()

        def sifunc(iren, ev):
            wsx, wsy = self.window.GetSize()
            ps = self.camera.GetParallelScale()
            newtxt = utils.precision(ps/wsy*wsx*length*dd,3)
            if units:
                newtxt += ' '+units
            if rlabel.GetText() != newtxt:
                rlabel.SetText(newtxt)

        self.renderer.AddActor(fractor)
        self.interactor.AddObserver('MouseWheelBackwardEvent', sifunc)
        self.interactor.AddObserver('MouseWheelForwardEvent', sifunc)
        self.interactor.AddObserver('InteractionEvent', sifunc)
        sifunc(0,0)
        return fractor


    def addCallback(self, eventName, func, priority=0.0):
        """
        Add a function to be executed while show() is active.
        Information about the event can be acquired with method getEvent().

        Return a unique id for the callback.

        The callback function (see example below) exposes a dictionary
        with the following information:

            - name: event name,
            - id: event unique identifier,
            - priority: event priority (float),
            - interactor: the interactor object,
            - at: renderer nr. where the event occured
            - actor: object picked by the mouse
            - picked3d: point picked in world coordinates
            - keyPressed: key pressed as string
            - picked2d: screen coords of the mouse pointer
            - delta2d: shift wrt previous position (to calculate speed, direction)
            - delta3d: ...same but in 3D world coords
            - angle2d: angle of mouse movement on screen
            - speed2d: speed of mouse movement on screen
            - speed3d: speed of picked point in world coordinates
            - isPoints: True if of class
            - isMesh: True if of class
            - isAssembly: True if of class
            - isVolume: True if of class Volume
            - isPicture: True if of class

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

        Example:
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

        .. hint:: examples/advanced/spline_draw.py, examples/basic/colorlines.py, ...
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
        vedo.logger.debug(f'registering event: {eventName} with id={cid}')
        return cid

    def removeCallback(self, cid):
        """
        Remove a callback function by its id
        or a whole category of callbacks by their name.

        Parameters
        ----------
        cid : int, str
            Unique id of the callback. If an event name is passed all callbacks of that type are removed.
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
            Either "create" or "destroy"

        timerId : int
            When destroying the timer, the ID of the timer as returned when created

        dt : int
            time in milliseconds between each repeated call

        oneShot: bool
            create a one shot timer of prescribed duration instead of a repeating one

        .. hint:: examples/advanced/timer_callback1.py, examples/advanced/timer_callback2.py
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
            e = "in plotter.timer(). Cannot understand action:\n"
            e+= "                          allowed actions: [create, destroy]"
            vedo.logger.error(e)
        return self


    def computeWorldPosition(self, pos2d, at=None, objs=(), bounds=(),
                             offset=None, pixeltol=None, worldtol=None,
        ):
        """
        Transform a 2D point on the screen into a 3D point inside the rendering scene.
        If a set of meshes is passed then points are placed onto these.

        Parameters
        ----------
        pos2d : list
            2D screen coordinates point.

        at : int, optional
            renderer number.

        objs : list, optional
            list of Mesh objects to project the point onto.

        bounds : list, optional
            specify a bounding box as [xmin,xmax, ymin,ymax, zmin,zmax].

        offset : float, optional
            specify an offset value.

        pixeltol : int, optional
            screen tolerance in pixels.

        worldtol : float, optional
            world coordinates tolerance.

        Returns
        -------
        numpy array
            the point in 3D world coordinates.

        .. hint:: examples/basic/cutFreeHand.py, examples/basic/mousehover3.py
        """
        if at is not None:
            renderer = self.renderers[at]
        else:
            renderer = self.renderer
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
        return np.array(worldPos)


    def _scan_input(self, wannabeacts):
        # scan the input of show
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
                    for sha in a.shadows:
                        if sha not in self.actors:
                            scannedacts.append(sha)
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
                            scannedacts.append(vedo.Mesh(utils.meshlab2vedo(a.mesh(i))))
                else:
                    scannedacts.append(vedo.Mesh(utils.meshlab2vedo(a)))

            else:
                vedo.logger.error(f"cannot understand input in show(): {type(a)}")
        return scannedacts


    def show(self,
             *actors,
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
             mode=0,
             rate=None,
             bg=None,
             bg2=None,
             size=None,
             title=None,
             q=False,
        ):
        """
        Render a list of actors.

        Parameters
        ----------
        at : int, optional
            number of the renderer to plot to, in case of more than one exists

        shape : list, str, optional
            Number of sub-render windows inside of the main window. E.g.:
            specify two across with shape=(2,1) and a two by two grid
            with shape=(2, 2).  By default there is only one renderer.

            Can also accept a shape as string descriptor. E.g.:

            - shape="3|1" means 3 plots on the left and 1 on the right,
            - shape="4/2" means 4 plots on top of 2 at bottom.

        axes : int, optional
            set the type of axes to be shown:

            - 0,  no axes
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

            Axis type-1 can be fully customized by passing a dictionary.
            Check addons.Axes() for the full list of options.

        azimuth/elevation/roll : float, optional
            move camera accordingly the specified value

        viewup: str, list
            either ['x', 'y', 'z'] or a vector to set vertical direction

        resetcam : bool
            re-adjust camera position to fit objects

        camera : dict, vtkCamera
            camera parameters can further be specified with a dictionary
            assigned to the ``camera`` keyword (E.g. `show(camera={'pos':(1,2,3), 'thickness':1000,})`)

            - pos, `(list)`,  the position of the camera in world coordinates

            - focalPoint `(list)`, the focal point of the camera in world coordinates

            - viewup `(list)`, the view up direction for the camera

            - distance `(float)`, set the focal point to the specified distance from the camera position.

            - clippingRange `(float)`, distance of the near and far clipping planes along the direction of projection.

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

        interactive : bool
            pause and interact with window (True) or continue execution (False)

        rate : float
            maximum rate of `show()` in Hertz
        mode : int, str
            set the type of interaction

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

        q : bool
            force program to quit after `show()` command returns.
        """
        if self.wxWidget:
            return self

        if len(self.renderers): # in case of notebooks

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
            if self.size[0] == 'f':  # full screen
                self.size = 'fullscreen'
                self.window.SetFullScreen(True)
                self.window.BordersOn()
            else:
                self.window.SetSize(int(self.size[0]), int(self.size[1]))

        if not vedo.notebookBackend:
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
            self._interactive = False

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
        if vedo.notebookBackend:
            if vedo.notebookBackend in ['k3d', 'ipygany', 'itkwidgets']:
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
            return self

        if interactive is not None:
            self._interactive = interactive

        if self.interactor:
            if not self.interactor.GetInitialized():
                self.interactor.Initialize()
                self.interactor.RemoveObservers("CharEvent")

        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetParallelProjection(settings.useParallelProjection)
        if self.sharecam:
            for r in self.renderers:
                r.SetActiveCamera(self.camera)

        if self.qtWidget is not None:
            self.qtWidget.GetRenderWindow().AddRenderer(self.renderer)

        if len(self.renderers) == 1:
            self.renderer.SetActiveCamera(self.camera)

        if vedo.vtk_version[0] == 9 and "Darwin" in vedo.sys_platform:
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
                    and not (vedo.vtk_version[0] == 9 and "Linux" in vedo.sys_platform)  # Linux vtk9 is bugged
                    ):
                    #check balloons
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
        if vedo.notebookBackend in ["panel","ipyvtk"]:
            return backends.getNotebookBackend(0, 0, 0)
        #########################################################################

        if self.resetcam:
            self.renderer.ResetCamera()

        if len(self.renderers) > 1:
            self.frames = self.addRendererFrame()

        if self.flagWidget:
            self.flagWidget.EnabledOn()

        if zoom:
            if zoom == "tight":
                self.resetCamera(tight=0.04)
            elif zoom == "tightest":
                self.resetCamera(tight=0.0001)
            else:
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
                mode = 12

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
                vedo.logger.warning(f"in show(cam=...), key(s) not recognized: {camera.keys()}")
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
        if vedo.notebookBackend == "2d":
            return backends.getNotebookBackend(0, 0, 0)
        #########################################################################


        if self.interactor: # can be offscreen..

            if settings.allowInteraction:
                self.allowInteraction()

            # Set the style of interaction
            # see https://vtk.org/doc/nightly/html/classvtkInteractorStyle.html
            if mode == 0 or mode == "TrackballCamera":
                #csty = self.interactor.GetInteractorStyle().GetCurrentStyle().GetClassName()
                #if "TrackballCamera" not in csty:
                # this causes problems (when pressing 3 eg) :
                if self.qtWidget:
                    self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            elif mode == 1 or mode == "TrackballActor":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
            elif mode == 2 or mode == "JoystickCamera":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleJoystickCamera())
            elif mode == 3 or mode == "JoystickActor":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleJoystickActor())
            elif mode == 4 or mode == "Flight":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleFlight())
            elif mode == 5 or mode == "RubberBand2D":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBand2D())
            elif mode == 6 or mode == "RubberBand3D":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBand3D())
            elif mode == 7 or mode == "RubberBandZoom":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandZoom())
            elif mode == 8 or mode == "Context":
                self.interactor.SetInteractorStyle(vtk.vtkContextInteractorStyle())
            elif mode == 9 or mode == "3D":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyle3D())
            elif mode ==10 or mode == "Terrain":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTerrain())
            elif mode ==11 or mode == "Unicam":
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleUnicam())
            elif mode ==12 or mode == "Image" or mode == "image":
                astyle = vtk.vtkInteractorStyleImage()
                astyle.SetInteractionModeToImage3D()
                self.interactor.SetInteractorStyle(astyle)

            if self._interactive:
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

        Parameters
        ----------
        at : int
            specify the renderer number

        pos : list
            icon position in the range [1-4] indicating one of the 4 corners,
            or it can be a tuple (x,y) as a fraction of the renderer size.

        size : float
            size of the square inset

        draggable : bool
            if True the subrenderer space can be dragged around

        c : color
            color of the inset frame when dragged

        .. hint:: [inset.py](https://github.com/marcomusy/vedo/tree/master/examples/other/inset.py)
            .. image:: https://user-images.githubusercontent.com/32848391/56758560-3c3f1300-6797-11e9-9b33-49f5a4876039.jpg
        """
        if not self.interactor:
            return None
        pos = options.pop("pos", 0)
        size = options.pop("size", 0.1)
        c = options.pop("c", 'lb')
        at = options.pop("at", None)
        draggable = options.pop("draggable", True)

        if not self.renderer:
            vedo.logger.warning("call addInset() only after first rendering of the scene.")
            save_int = self._interactive
            self.show(interactive=0)
            self._interactive = save_int
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
            renderer = self.renderers[at]
        else:
            renderer = self.renderer
        if not renderer:
            return self

        if actors is None:
            renderer.RemoveAllViewProps()
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
        return self


    def closeWindow(self):
        """Close the current or the input rendering window.

        .. hint:: examples/basic/closewindow.py
        """
        for r in self.renderers:
            r.RemoveAllObservers()
        if hasattr(self, 'window') and self.window:
            self.window.Finalize()
            if hasattr(self, 'interactor') and self.interactor:
                self.interactor.ExitCallback()
                try:
                    self.interactor.SetDone(True)
                except AttributeError:
                    pass
                self.interactor.TerminateApp()
                self.interactor = None
            self.window = None
        return self

    def close(self):
        """Close the Plotter instance and release resources."""
        self.closeWindow()
        self.actors = []
        vedo.plotter_instance = None

    def screenshot(self, filename='screenshot.png', scale=None, asarray=False):
        """Take a screenshot of the Plotter window.

        Parameters
        ----------
        scale : int
            set image magnification as an integer multiplicating factor

        asarray : bool
            return a numpy array of the image instead of writing a file
        """
        retval = vedo.io.screenshot(filename, scale, asarray)
        return retval

    def topicture(self, scale=None):
        """Generate a Picture object from the current rendering window.

        Parameters
        ----------
        scale : int
            set image magnification as an integer multiplicating factor
        """
        if scale is None:
            scale = settings.screeshotScale

        if settings.screeshotLargeImage:
           w2if = vtk.vtkRenderLargeImage()
           w2if.SetInput(self.renderer)
           w2if.SetMagnification(scale)
        else:
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(self.window)
            if hasattr(w2if, 'SetScale'):
                w2if.SetScale(scale, scale)
            if settings.screenshotTransparentBackground:
                w2if.SetInputBufferTypeToRGBA()
            w2if.ReadFrontBufferOff()  # read from the back buffer
        w2if.Update()
        return vedo.picture.Picture(w2if.GetOutput())

    def export(self, filename='scene.npz', binary=False):
        """Export scene to file to HTML, X3D or Numpy file.

        .. hint:: examples/other/export_x3d.py, examples/other/export_numpy.py
        """
        vedo.io.exportWindow(filename, binary=binary)
        return self

    def colorPicker(self, xy, verbose=False):
        """Pick color of specific (x,y) pixel on the screen."""
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.window)
        w2if.ReadFrontBufferOff()
        w2if.Update()
        nx, ny = self.window.GetSize()
        varr = w2if.GetOutput().GetPointData().GetScalars()

        arr = utils.vtk2numpy(varr).reshape(ny,nx,3)
        x,y = int(xy[0]), int(xy[1])
        if y < ny and  x < nx:

            rgb = arr[y,x]

            if verbose:
                vedo.printc('Pixel', [x,y], 'has RGB[',  end='')
                vedo.printc('█', c=[rgb[0],0,0], end='')
                vedo.printc('█', c=[0,rgb[1],0], end='')
                vedo.printc('█', c=[0,0,rgb[2]], end='')
                vedo.printc('] = ', end='')
                cnm = vedo.getColorName(rgb)
                if np.sum(rgb) < 150:
                    vedo.printc(rgb.tolist(), vedo.colors.rgb2hex(np.array(rgb)/255), c='w',
                           bc=rgb, invert=1, end='')
                    vedo.printc('  -> '+cnm, invert=1, c='w')
                else:
                    vedo.printc(rgb.tolist(), vedo.colors.rgb2hex(np.array(rgb)/255), c=rgb, end='')
                    vedo.printc('  -> '+cnm, c=cnm)

            return rgb

        return None


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

        # utils.vedo.printc('Pressed key:', self.keyheld, key, c='y', box='-')

        if key in ["Shift_L", "Control_L", "Super_L", "Alt_L",
                   "Shift_R", "Control_R", "Super_R", "Alt_R", "Menu"]:
            self.keyheld = key

        if key in ["q", "space", "Return", "F12"]:
            iren.ExitCallback()
            return

        elif key == "Escape":
            vedo.logger.info("Closing window now. Plotter.escaped is set to True.")
            self.escaped = True # window will be escaped ASAP
            iren.ExitCallback()
            return

        elif key == "F1":
            vedo.logger.info("Execution aborted. Exiting python kernel now.")
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

        elif key == "u":
            pval = self.renderer.GetActiveCamera().GetParallelProjection()
            self.renderer.GetActiveCamera().SetParallelProjection(not pval)
            if pval:
                self.renderer.ResetCamera()

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
            msg  = " ==========================================================\n"
            msg += "| Press: i     print info about selected object            |\n"
            msg += "|        I     print the RGB color under the mouse         |\n"
            msg += "|        <-->  use arrows to reduce/increase opacity       |\n"
            msg += "|        w/s   toggle wireframe/surface style              |\n"
            msg += "|        p/P   change point size of vertices               |\n"
            msg += "|        l     toggle edges visibility                     |\n"
            msg += "|        x     toggle mesh visibility                      |\n"
            msg += "|        X     invoke a cutter widget tool                 |\n"
            msg += "|        1-3   change mesh color                           |\n"
            msg += "|        4     use data array as colors, if present        |\n"
            msg += "|        5-6   change background color(s)                  |\n"
            msg += "|        09+-  (on keypad) or +/- to cycle axes style      |\n"
            msg += "|        k     cycle available lighting styles             |\n"
            msg += "|        K     cycle available shading styles              |\n"
            msg += "|        A     toggle anti-aliasing                        |\n"
            msg += "|        D     toggle depth-peeling (for transparencies)   |\n"
            msg += "|        o/O   add/remove light to scene and rotate it     |\n"
            msg += "|        n     show surface mesh normals                   |\n"
            msg += "|        a     toggle interaction to Actor Mode            |\n"
            msg += "|        j     toggle interaction to Joystick Mode         |\n"
            msg += "|        u     toggle perspective/parallel projection      |\n"
            msg += "|        r     reset camera position                       |\n"
            msg += "|        C     print current camera settings               |\n"
            msg += "|        S     save a screenshot                           |\n"
            msg += "|        E/F   export 3D scene to numpy file or X3D        |\n"
            msg += "|        q     return control to python script             |\n"
            msg += "|        Esc   abort execution and exit python kernel      |\n"
            msg += "|----------------------------------------------------------|\n"
            msg += "| Mouse: Left-click    rotate scene / pick actors          |\n"
            msg += "|        Middle-click  pan scene                           |\n"
            msg += "|        Right-click   zoom scene in or out                |\n"
            msg += "|        Cntrl-click   rotate scene                        |\n"
            msg += "|----------------------------------------------------------|\n"
            msg += "| Check out documentation at:  https://vedo.embl.es        |\n"
            msg += " =========================================================="
            vedo.printc(msg, dim=1)

            msg = " vedo " + vedo.__version__ + " "
            vedo.printc(msg, invert=1, dim=1, end="")
            vtkVers = vtk.vtkVersion().GetVTKVersion()
            msg = "| vtk " + str(vtkVers)
            msg += " | python " + str(sys.version_info[0]) + "." + str(sys.version_info[1])
            vedo.printc(msg, invert=0, dim=1)
            return

        elif key == "a":
            iren.ExitCallback()
            cur = iren.GetInteractorStyle()
            if isinstance(cur, vtk.vtkInteractorStyleTrackballCamera):
                msg = "\nInteractor style changed to TrackballActor\n"
                msg +="  you can now move and rotate individual meshes:\n"
                msg +="  press X twice to save the repositioned mesh\n"
                msg +="  press 'a' to go back to normal style"
                vedo.printc(msg)
                iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
            else:
                iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            iren.Start()
            return

        elif key == "A": # toggle antialiasing
            msam = self.window.GetMultiSamples()
            if not msam:
                self.window.SetMultiSamples(8)
            else:
                self.window.SetMultiSamples(0)
            msam = self.window.GetMultiSamples()
            if msam:
                vedo.printc(f'Antialiasing is now set to {msam} samples', c=bool(msam))
            else:
                vedo.printc('Antialiasing is now disabled', c=bool(msam))

        elif key == "D": # toggle depthpeeling
            udp = not self.renderer.GetUseDepthPeeling()
            self.renderer.SetUseDepthPeeling(udp)
            #self.renderer.SetUseDepthPeelingForVolumes(udp)
            # print(self.window.GetAlphaBitPlanes())
            if udp:
                self.window.SetAlphaBitPlanes(1)
                self.renderer.SetMaximumNumberOfPeels(settings.maxNumberOfPeels)
                self.renderer.SetOcclusionRatio(settings.occlusionRatio)
            self.interactor.Render()
            wasUsed = self.renderer.GetLastRenderingUsedDepthPeeling()
            rnr = self.renderers.index(self.renderer)
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
            vedo.printc('           clippingRange='+utils.precision(cam.GetClippingRange(),4)+',', c='y')
            vedo.printc(')', c='y')
            vedo.printc('show(mymeshes, camera=cam)', c='y')
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
                pal = vedo.colors.palettes[settings.palette % len(vedo.colors.palettes)]
                self.clickedActor.GetProperty().SetColor(pal[(self._icol) % 10])

        elif key == "2":
            self._icol += 1
            settings.palette += 1
            settings.palette = settings.palette % len(vedo.colors.palettes)
            if isinstance(self.clickedActor, vedo.Points):
                self.clickedActor.GetMapper().ScalarVisibilityOff()
                pal = vedo.colors.palettes[settings.palette % len(vedo.colors.palettes)]
                self.clickedActor.GetProperty().SetColor(pal[(self._icol) % 10])

        elif key == "3":
            bsc = ['b5','cyan5', 'g4', 'o5', 'p5', 'r4', 'teal4', 'yellow4']
            self._icol += 1
            if isinstance(self.clickedActor, vedo.Points):
                self.clickedActor.GetMapper().ScalarVisibilityOff()
                self.clickedActor.GetProperty().SetColor(vedo.getColor(bsc[(self._icol) % len(bsc)]))

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
            self.interactor.Start()

        elif key == "F":
            vedo.io.exportWindow('scene.x3d')
            vedo.printc("Try: firefox scene.html", c="blue")

        elif key == "i":  # print info
            if self.clickedActor:
                utils.printInfo(self.clickedActor)
            else:
                utils.printInfo(self)

        elif key == "I":  # print color under the mouse
            x, y = iren.GetEventPosition()
            self.colorPicker([x,y], verbose=True)

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
