from __future__ import division, print_function
import time
import sys
import vtk
import numpy as np

from vtkplotter import __version__
import vtkplotter.vtkio as vtkio
import vtkplotter.utils as utils
import vtkplotter.colors as colors
from vtkplotter.assembly import Assembly
from vtkplotter.mesh import Mesh
from vtkplotter.picture import Picture
from vtkplotter.volume import Volume
import vtkplotter.docs as docs
import vtkplotter.settings as settings
import vtkplotter.addons as addons
import vtkplotter.backends as backends

__doc__ = (
    """
Defines main class ``Plotter`` to manage actors and 3D rendering.
"""
    + docs._defs
)

__all__ = ["show", "clear", "Plotter", "closeWindow", "closePlotter", "interactive"]

########################################################################
def show(*actors, **options):
    """
    Create on the fly an instance of class ``Plotter`` and show the object(s) provided.

    Allowed input objects are: ``filename``, ``vtkPolyData``, ``vtkActor``,
    ``vtkActor2D``, ``vtkImageActor``, ``vtkAssembly`` or ``vtkVolume``.

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

          - 0,  no axes,
          - 1,  draw three gray grid walls
          - 2,  show cartesian axes from (0,0,0)
          - 3,  show positive range of cartesian axes from (0,0,0)
          - 4,  show a triad at bottom left
          - 5,  show a cube at bottom left
          - 6,  mark the corners of the bounding box
          - 7,  draw a simple ruler at the bottom of the window
          - 8,  show the ``vtkCubeAxesActor`` object,
          - 9,  show the bounding box outLine,
          - 10, show three circles representing the maximum bounding box
          - 11, show a large grid on the x-y plane (use with zoom=8)
          - 12, show polar axes

        Axis type-1 can be fully customized by passing a dictionary ``axes=dict()`` where:

            - `xtitle`,            ['x'], x-axis title text.
            - `ytitle`,            ['y'], y-axis title text.
            - `ztitle`,            ['z'], z-axis title text.
            - `numberOfDivisions`, [automatic], number of divisions on the shortest axis
            - `axesLineWidth`,       [1], width of the axes lines
            - `gridLineWidth`,       [1], width of the grid lines
            - `reorientShortTitle`, [True], titles shorter than 2 letter are placed horizontally
            - `originMarkerSize`, [0.01], draw a small cube on the axis where the origin is
            - `titleDepth`,          [0], extrusion fractional depth of title text
            - `xyGrid`,           [True], show a gridded wall on plane xy
            - `yzGrid`,           [True], show a gridded wall on plane yz
            - `zxGrid`,           [True], show a gridded wall on plane zx
            - `zxGrid2`,         [False], show zx plane on opposite side of the bounding box
            - `xyGridTransparent`  [False], make grid plane completely transparent
            - `xyGrid2Transparent` [False], make grid plane completely transparent on opposite side box
            - `xyPlaneColor`,   ['gray'], color of the plane
            - `xyGridColor`,    ['gray'], grid line color
            - `xyAlpha`,          [0.15], grid plane opacity
            - `showTicks`,        [True], show major ticks
            - `xTitlePosition`,   [0.32], title fractional positions along axis
            - `xTitleOffset`,     [0.05], title fractional offset distance from axis line
            - `xTitleJustify`, ["top-right"], title justification
            - `xTitleRotation`,      [0], add a rotation of the axis title
            - `xLineColor`,  [automatic], color of the x-axis
            - `xTitleColor`, [automatic], color of the axis title
            - `xTitleBackfaceColor`, [None],  color of axis title on its backface
            - `xTitleSize`,      [0.025], size of the axis title
            - `xHighlightZero`,   [True], draw a line highlighting zero position if in range
            - `xHighlightZeroColor`, [automatic], color of the line highlighting the zero position
            - `xTickRadius`,     [0.005], radius of the major ticks
            - `xTickThickness`, [0.0025], thickness of the major ticks along their axis
            - `xTickColor`,  [automatic], color of major ticks
            - `xMinorTicks`,         [1], number of minor ticks between two major ticks
            - `tipSize`,          [0.01], size of the arrow tip
            - `xPositionsAndLabels`   [], assign custom tick positions and labels [(pos1, label1), ...]
            - `xLabelPrecision`,     [2], nr. of significative digits to be shown
            - `xLabelSize`,      [0.015], size of the numeric labels along axis
            - `xLabelOffset`,    [0.025], offset of numeric labels

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
    :param int interactorStyle: set the type of interaction
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
        -10 = Terrain
        -11 = Unicam

    :param bool q:  force program to quit after `show()` command returns.

    :param bool newPlotter: if set to `True`, a call to ``show`` will instantiate
        a new ``Plotter`` object (a new window) instead of reusing the first created.

        See e.g.: |readVolumeAsIsoSurface.py|_

    :return: the current ``Plotter`` class instance.

    .. note:: With multiple renderers, keyword ``at`` can become a `list`, e.g.

        .. code-block:: python

            from vtkplotter import *
            s = Sphere()
            c = Cube()
            p = Paraboloid()
            show(s, c, at=[0, 1], shape=(3,1))
            show(p, at=2, interactive=True)
            #
            # is equivalent to:
            vp = Plotter(shape=(3,1))
            s = Sphere()
            c = Cube()
            p = Paraboloid()
            vp.show(s, at=0)
            vp.show(p, at=1)
            vp.show(c, at=2, interactive=True)
    """
    at = options.pop("at", None)
    shape = options.pop("shape", (1, 1))
    N = options.pop("N", None)
    pos = options.pop("pos", (0, 0))
    size = options.pop("size", "auto")
    screensize = options.pop("screensize", "auto")
    title = options.pop("title", "")
    #xtitle = options.pop("xtitle", "x")
    #ytitle = options.pop("ytitle", "y")
    #ztitle = options.pop("ztitle", "z")
    bg = options.pop("bg", "blackboard")
    bg2 = options.pop("bg2", None)
    axes = options.pop("axes", 4)
    verbose = options.pop("verbose", True)
    interactive = options.pop("interactive", None)
    offscreen = options.pop("offscreen", False)
    sharecam = options.pop("sharecam", True)
    resetcam = options.pop("resetcam", True)
    zoom = options.pop("zoom", None)
    viewup = options.pop("viewup", "")
    azimuth = options.pop("azimuth", 0)
    elevation = options.pop("elevation", 0)
    roll = options.pop("roll", 0)
    camera = options.pop("camera", None)
    interactorStyle = options.pop("interactorStyle", 0)
    newPlotter = options.pop("newPlotter", False)
    q = options.pop("q", False)

    if len(options):
        for op in options:
            colors.printc("Warning: unknown keyword in show():", op, c=5)

    if len(actors) == 0:
        actors = None
    elif len(actors) == 1:
        actors = actors[0]
    else:
        actors = utils.flatten(actors)

    if settings.plotter_instance and newPlotter is False:
        vp = settings.plotter_instance
        #vp.renderer.SetBackground(colors.getColor(bg))
    else:
        if utils.isSequence(at):
            if not utils.isSequence(actors):
                colors.printc("~times show() Error: input must be a list.", c=1)
                raise RuntimeError()
            if len(at) != len(actors):
                colors.printc("~times show() Error: lists 'input' and 'at', must have equal lengths.", c=1)
                raise RuntimeError()
            if len(at) > 1 and (shape == (1, 1) and N is None):
                N = max(at) + 1
        elif at is None and (N or shape != (1, 1)):
            if not utils.isSequence(actors):
                colors.printc('~times show() Error: N or shape is set, but input is not a sequence.', c=1)
                colors.printc('              you may need to specify e.g. at=0', c=1)
                raise RuntimeError()
            at = range(len(actors))

        vp = Plotter(
            shape=shape,
            N=N,
            pos=pos,
            size=size,
            screensize=screensize,
            title=title,
            bg=bg,
            bg2=bg2,
            axes=axes,
            sharecam=sharecam,
            verbose=verbose,
            interactive=interactive,
            offscreen=offscreen,
        )
        #vp.xtitle = xtitle
        #vp.ytitle = ytitle
        #vp.ztitle = ztitle

    # use _vp_to_return because vp.show() can return a k3d/panel plot
    if utils.isSequence(at):
        for i, a in enumerate(actors):
            _vp_to_return = vp.show(
                a,
                at=i,
                zoom=zoom,
                resetcam=resetcam,
                viewup=viewup,
                azimuth=azimuth,
                elevation=elevation,
                roll=roll,
                camera=camera,
                interactive=interactive,
                interactorStyle=interactorStyle,
                q=q,
            )
        vp.interactor.Start()
    else:
        _vp_to_return = vp.show(
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
            q=q,
        )

    return _vp_to_return


def interactive():
    """Go back to the rendering window interaction mode."""
    if settings.plotter_instance:
        if hasattr(settings.plotter_instance, 'interactor'):
            if settings.plotter_instance.interactor:
                settings.plotter_instance.interactor.Start()
    return settings.plotter_instance


def clear(actor=()):
    """
    Clear specific actor or list of actors from the current rendering window.
    """
    if not settings.plotter_instance:
        return
    settings.plotter_instance.clear(actor)
    return settings.plotter_instance


def closeWindow(plotterInstance=None):
    """Close the current or the input rendering window."""
    if not plotterInstance:
        from vtkplotter.settings import plotter_instance
        plotterInstance  = plotter_instance
        if not plotterInstance:
            return
    if plotterInstance.interactor:
        plotterInstance.interactor.ExitCallback()
        plotterInstance.closeWindow()
    return plotterInstance


def closePlotter():
    """Close the current instance of ``Plotter`` and its rendering window."""
    if settings.plotter_instance:
        settings.plotter_instance.closeWindow()
        settings.plotter_instance = None
        settings.plotter_instances = []
        settings.collectable_actors = []
    return None


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
      - 7,  draw a simple ruler at the bottom of the window
      - 8,  show the ``vtkCubeAxesActor`` object
      - 9,  show the bounding box outLine,
      - 10, show three circles representing the maximum bounding box,
      - 11, show a large grid on the x-y plane (use with zoom=8)
      - 12, show polar axes.

    Axis type-1 can be fully customized by passing a dictionary ``axes=dict()`` where:

        - `xtitle`,            ['x'], x-axis title text.
        - `ytitle`,            ['y'], y-axis title text.
        - `ztitle`,            ['z'], z-axis title text.
        - `numberOfDivisions`, [automatic], number of divisions on the longest axis
        - `axesLineWidth`,       [1], width of the axes lines
        - `gridLineWidth`,       [1], width of the grid lines
        - `reorientShortTitle`, [True], titles shorter than 2 letter are placed horizontally
        - `originMarkerSize`, [0.01], draw a small cube on the axis where the origin is
        - `enableLastLabel`, [False], show last numeric label on axes
        - `titleDepth`,          [0], extrusion fractional depth of title text
        - `xyGrid`,           [True], show a gridded wall on plane xy
        - `yzGrid`,           [True], show a gridded wall on plane yz
        - `zxGrid`,           [True], show a gridded wall on plane zx
        - `zxGrid2`,         [False], show zx plane on opposite side of the bounding box
        - `xyGridTransparent`  [False], make grid plane completely transparent
        - `xyGrid2Transparent` [False], make grid plane completely transparent on opposite side box
        - `xyPlaneColor`,   ['gray'], color of the plane
        - `xyGridColor`,    ['gray'], grid line color
        - `xyAlpha`,          [0.15], grid plane opacity
        - `showTicks`,        [True], show major ticks
        - `xTitlePosition`,   [0.32], title fractional positions along axis
        - `xTitleOffset`,     [0.05], title fractional offset distance from axis line
        - `xTitleJustify`, ["top-right"], title justification
        - `xTitleRotation`,      [0], add a rotation of the axis title
        - `xLineColor`,  [automatic], color of the x-axis
        - `xTitleColor`, [automatic], color of the axis title
        - `xTitleBackfaceColor`, [None],  color of axis title on its backface
        - `xTitleSize`,      [0.025], size of the axis title
        - `xHighlightZero`,   [True], draw a line highlighting zero position if in range
        - `xHighlightZeroColor`, [automatic], color of the line highlighting the zero position
        - `xTickRadius`,     [0.005], radius of the major ticks
        - `xTickThickness`, [0.0025], thickness of the major ticks along their axis
        - `xTickColor`,  [automatic], color of major ticks
        - `xMinorTicks`,         [1], number of minor ticks between two major ticks
        - `tipSize`,          [0.01], size of the arrow tip
        - `xLabelPrecision`,     [2], nr. of significative digits to be shown
        - `xLabelSize`,      [0.015], size of the numeric labels along axis
        - `xLabelOffset`,    [0.025], offset of numeric labels

    :param bool sharecam: if False each renderer will have an independent vtkCamera
    :param bool interactive: if True will stop after show() to allow interaction w/ window
    :param bool offscreen: if True will not show the rendering window
    :param QVTKRenderWindowInteractor qtWidget:

      render in a Qt-Widget using an QVTKRenderWindowInteractor.
      Overrides offscreen to True
      Overrides interactive to False
      Sets setting.usingQt to True
      See Also: example qt_windows.py

    |multiwindows|
    """

    def __init__(
        self,
        shape=(1, 1),
        N=None,
        pos=(0, 0),
        size="auto",
        screensize="auto",
        title="",
        bg="blackboard",
        bg2=None,
        axes=4,
        sharecam=True,
        verbose=True,
        interactive=None,
        offscreen=False,
        qtWidget = None
    ):

        settings.plotter_instance = self
        settings.plotter_instances.append(self)

        if qtWidget is not None:
            # overrides the interactive and offscreen properties
            interactive = False
            offscreen = True
            settings.usingQt = True

        if interactive is None:
            if N or shape != (1, 1):
                interactive = False
            else:
                interactive = True

        if not interactive:
            verbose = False

        self.verbose = verbose
        self.actors = []  # list of actors to be shown
        self.clickedActor = None  # holds the actor that has been clicked
        self.renderer = None  # current renderer
        self.renderers = []  # list of renderers
        self.pos = pos
        self.shape = shape  # don't remove this line
        self.interactive = interactive  # allows to interact with renderer
        self.axes = axes  # show axes type nr.
        self.title = title  # window title
        self.sharecam = sharecam  # share the same camera if multiple renderers
        self._legend = []  # list of legend entries for actors
        self.legendSize = 0.15  # size of legend
        self.legendBC = (0.96, 0.96, 0.9)  # legend background color
        self.legendPos = 2  # 1=topright, 2=top-right, 3=bottom-left
        self.picked3d = None  # 3d coords of a clicked point on an actor
        self.backgrcol = bg
        self.offscreen = offscreen
        self.qtWidget = qtWidget # (QVTKRenderWindowInteractor)

        self.flagWidget = None
        self._flagRep = None

        # mostly internal stuff:
        self.justremoved = None
        self.axes_instances = []
        self.icol = 0
        self.clock = 0
        self._clockt0 = time.time()
        self.initializedPlotter = False
        self.initializedIren = False
        self.keyPressFunction = None
        self.sliders = []
        self.buttons = []
        self.widgets = []
        self.scalarbars = []
        self.cutterWidget = None
        self.backgroundRenderer = None
        self.mouseLeftClickFunction = None
        self.mouseMiddleClickFunction = None
        self.mouseRightClickFunction = None
        self._first_viewup = True
        self.extralight = None
        self.size = size
        self.interactor = None

        self.xtitle = settings.xtitle  # x axis label and units
        self.ytitle = settings.ytitle  # y axis label and units
        self.ztitle = settings.ztitle  # z axis label and units

        # build the rendering window:
        if settings.useOpenVR:
            self.camera = vtk.vtkOpenVRCamera()
            self.window =vtk.vtkOpenVRRenderWindow()
        else:
            self.camera = vtk.vtkCamera()
            self.window = vtk.vtkRenderWindow()

        ############################################################
        notebookBackend = settings.notebookBackend
        if notebookBackend and notebookBackend.lower() == '2d':
            self.offscreen = True
            if self.size == "auto":
                self.size = (900, 700)
        if (notebookBackend
            and notebookBackend != "panel"
            and notebookBackend.lower() != "2d"):
            self.interactive = False
            self.interactor = None
            self.window = None
            self.camera = None # let the backend choose
            if self.size == "auto":
                self.size = (1000, 1000)
            ########################################################
            return #################################################
            ########################################################

        # more settings
        if settings.useDepthPeeling:
            self.window.SetAlphaBitPlanes(settings.alphaBitPlanes)
            self.window.SetMultiSamples(settings.multiSamples)

        self.window.SetPolygonSmoothing(settings.polygonSmoothing)
        self.window.SetLineSmoothing(settings.lineSmoothing)
        self.window.SetPointSmoothing(settings.pointSmoothing)

        # sort out screen size
        if screensize == "auto":
            aus = self.window.GetScreenSize()
            if aus and len(aus) == 2 and aus[0] > 100 and aus[1] > 100:  # seems ok
                if aus[0] / aus[1] > 2:  # looks like there are 2 or more screens
                    screensize = (int(aus[0] / 2), aus[1])
                else:
                    screensize = aus
            else:  # it went wrong, use a default 1.5 ratio
                screensize = (2160, 1440)

        x, y = screensize
        if N:  # N = number of renderers. Find out the best
            if shape != (1, 1):  # arrangement based on minimum nr. of empty renderers
                colors.printc("Warning: having set N, shape is ignored.", c=1)
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

        self.shape = shape

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
                if settings.useFXAA is not None:
                    r.SetUseFXAA(settings.useFXAA)
                    self.window.SetMultiSamples(settings.multiSamples)
                if settings.useDepthPeeling:
                    r.SetUseDepthPeeling(True)
                    r.SetMaximumNumberOfPeels(settings.maxNumberOfPeels)
                    r.SetOcclusionRatio(settings.occlusionRatio)
                r.SetBackground(colors.getColor(self.backgrcol))
                self.axes_instances.append(None)

            self.shape = (n+m,)

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

            ############################
            self.shape = shape

            if sum(shape) > 3:
                self.legendSize *= 2

            image_actor=None
            bgname = str(self.backgrcol).lower()
            if ".jpg" in bgname or ".jpeg" in bgname or ".png" in bgname:
                self.window.SetNumberOfLayers(2)
                self.backgroundRenderer = vtk.vtkRenderer()
                self.backgroundRenderer.SetLayer(0)
                self.backgroundRenderer.InteractiveOff()
                self.backgroundRenderer.SetBackground(colors.getColor(bg2))
                image_actor = Picture(self.backgrcol)
                self.window.AddRenderer(self.backgroundRenderer)
                self.backgroundRenderer.AddActor(image_actor)

            for i in reversed(range(shape[0])):
                for j in range(shape[1]):
                    if settings.useOpenVR:
                        arenderer = vtk.vtkOpenVRRenderer()
                    else:
                        arenderer = vtk.vtkRenderer()
                        arenderer.SetUseHiddenLineRemoval(settings.hiddenLineRemoval)
                        arenderer.SetLightFollowCamera(settings.lightFollowsCamera)
                        arenderer.SetUseFXAA(settings.useFXAA)
                        if settings.useFXAA:
                            self.window.SetMultiSamples(settings.multiSamples)
                        arenderer.SetUseDepthPeeling(settings.useDepthPeeling)

                    if image_actor:
                        arenderer.SetLayer(1)

                    arenderer.SetBackground(colors.getColor(self.backgrcol))
                    if bg2:
                        arenderer.GradientBackgroundOn()
                        arenderer.SetBackground2(colors.getColor(bg2))

                    x0 = i / shape[0]
                    y0 = j / shape[1]
                    x1 = (i + 1) / shape[0]
                    y1 = (j + 1) / shape[1]
                    arenderer.SetViewport(y0, x0, y1, x1)
                    self.renderers.append(arenderer)
                    self.axes_instances.append(None)

        if len(self.renderers):
            self.renderer = self.renderers[0]

        if self.size[0] == 'f':  # full screen
            self.size = 'fullscreen'
            self.window.SetFullScreen(True)
            self.window.BordersOn()
        else:
            self.window.SetSize(int(self.size[0]), int(self.size[1]))

        self.window.SetPosition(pos)

        if not title:
            title = " vtkplotter " + __version__ + ", vtk " + vtk.vtkVersion().GetVTKVersion()
            title += ", python " + str(sys.version_info[0]) + "." + str(sys.version_info[1])

        self.window.SetWindowName(title)

        if not settings.usingQt:
            for r in self.renderers:
                self.window.AddRenderer(r)

        if self.qtWidget is not None:
            self.interactor = self.qtWidget.GetRenderWindow().GetInteractor()
            self.window.SetOffScreenRendering(True)
            ########################
            return
            ########################

        if self.offscreen:
            if self.axes == 4 or self.axes == 5:
                self.axes = 0 #doesn't work with those
            self.window.SetOffScreenRendering(True)
            self.interactive = False
            self.interactor = None
            ########################
            return
            ########################

        if settings.notebookBackend == "panel":
            return

        if settings.useOpenVR:
            self.interactor = vtk.vtkOpenVRRenderWindowInteractor()
        else:
            self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(vsty)

        self.interactor.AddObserver("LeftButtonPressEvent", self._mouseleft)
        self.interactor.AddObserver("RightButtonPressEvent", self._mouseright)
        self.interactor.AddObserver("MiddleButtonPressEvent", self._mousemiddle)
        self.interactor.AddObserver("KeyPressEvent", self._keypress)

        if settings.allowInteraction:
            self._update_observer = None
            self._update_win_clock = time.time()

            def win_interact(iren, event):  # flushing renderer events
                if event == "TimerEvent":
                    iren.TerminateApp()

            self.interactor.AddObserver("TimerEvent", win_interact)

            def _allowInteraction():
                timenow = time.time()
                if timenow - self._update_win_clock > 0.1:
                    self._update_win_clock = timenow
                    self._update_observer = self.interactor.CreateRepeatingTimer(1)
                    if hasattr(self, 'interactor') and self.interactor:
                        self.interactor.Start()

                    if hasattr(self, 'interactor') and self.interactor:
                        # twice otherwise it crashes when pressing Esc (??)
                        self.interactor.DestroyTimer(self._update_observer)

            self.allowInteraction = _allowInteraction

        return
        ####################### ..init ends here.

    def __str__(self):
        utils.printInfo(self)
        return ""

    def __iadd__(self, actors):
        self.add(actors, render=False)
        return self

    def __isub__(self, actors):
        self.remove(actors)
        return self

    def add(self, actors, render=True):
        """Append input object to the internal list of actors to be shown.

        :return: returns input actor for possible concatenation.
        """
        if utils.isSequence(actors):
            for a in actors:
                if a not in self.actors:
                    self.actors.append(a)
                    if render and self.renderer:
                        self.renderer.AddActor(a)
            if render and self.interactor:
                self.interactor.Render()
            return None
        else:
            self.actors.append(actors)
            if render and self.renderer:
                self.renderer.AddActor(actors)
                if self.interactor:
                    self.interactor.Render()
            return actors

    def remove(self, actors, render=True):
        """Remove ``vtkActor`` or actor index from current renderer."""
        if not utils.isSequence(actors):
            actors = [actors]

        for a in actors:
            if self.renderer:
                self.renderer.RemoveActor(a)
                if hasattr(a, 'renderedAt'):
                    ir = self.renderers.index(self.renderer)
                    a.renderedAt.discard(ir)
                if hasattr(a, 'scalarbar') and a.scalarbar:
                    self.renderer.RemoveActor(a.scalarbar)
                if hasattr(a, 'trail') and a.trail:
                    self.renderer.RemoveActor(a.trail)
                    a.trailPoints = []
            if a in self.actors:
                i = self.actors.index(a)
                del self.actors[i]
        if render and hasattr(self, 'interactor') and self.interactor:
            self.interactor.Render()


    ####################################################
    def load(self, inputobj, c=None, alpha=1, threshold=False, spacing=(), unpack=True):
        """
        Load Mesh and Volume objects from file.
        The output will depend on the file extension. See examples below.

        :param c: color in RGB format, hex, symbol or name
        :param alpha: transparency (0=invisible)

        For volumetric data (tiff, slc, vti etc):
        :param float threshold: value to draw the isosurface, False by default to return a ``Volume``
        :param list spacing: specify the voxel spacing in the three dimensions
        :param bool unpack: only for multiblock data, if True returns a flat list of objects.

        :Example:
            .. code-block:: python

                from vtkplotter import datadir, load, show

                # Return an Mesh
                g = load(datadir+'ring.gmsh')
                show(g)

                # Return a list of 2 Mesh
                g = load([datadir+'250.vtk', datadir+'290.vtk'])
                show(g)

                # Return a list of meshes by reading all files in a directory
                # (if directory contains DICOM files then a Volume is returned)
                g = load(datadir+'timecourse1d/')
                show(g)

                # Return a Volume. Color/Opacity transfer function can be specified too.
                g = load(datadir+'embryo.slc')
                g.c(['y','lb','w']).alpha((0.0, 0.4, 0.9, 1))
                show(g)

                # Return a Mesh from a SLC volume with automatic thresholding
                g = load(datadir+'embryo.slc', threshold=True)
                show(g)
        """
        acts = vtkio.load(inputobj, c, alpha, threshold, spacing, unpack)
        if utils.isSequence(acts):
            self.actors += acts
        else:
            self.actors.append(acts)
        return acts


    def getVolumes(self, obj=None, renderer=None):
        """
        Return the list of the rendered Volumes.

        If ``obj`` is:
            ``None``, return volumes of current renderer

            ``int``, return volumes in given renderer number

        :param int,vtkRenderer renderer: specify which renederer to look into.
        """
        if renderer is None:
            renderer = self.renderer
        elif isinstance(renderer, int):
                renderer = self.renderers.index(renderer)
        else:
            return []

        if obj is None or isinstance(obj, int):
            if obj is None:
                acs = renderer.GetVolumes()
            elif obj >= len(self.renderers):
                colors.printc("~timesError in getVolumes: non existing renderer", obj, c=1)
                return []
            else:
                acs = self.renderers[obj].GetVolumes()
            vols = []
            acs.InitTraversal()
            for i in range(acs.GetNumberOfItems()):
                a = acs.GetNextItem()
                if a.GetPickable():
                    r = self.renderers.index(renderer)
                    if a == self.axes_instances[r]:
                        continue
                    vols.append(a)
            return vols

    def getActors(self, obj=None, renderer=None):
        """Obsolete. Please use getMeshes()"""
        colors.printc("getActors is obsolete, use getMeshes() instead.", box='=', c=1)
        raise RuntimeError

    def getMeshes(self, obj=None, renderer=None):
        """
        Return a list of Meshes (which may include Volume objects too).

        If ``obj`` is:
            ``None``, return meshes of current renderer

            ``int``, return meshes in given renderer number

            ``vtkAssembly`` return the contained meshes

            ``string``, return meshes matching legend name

        :param int,vtkRenderer renderer: specify which renederer to look into.
        """
        if renderer is None:
            renderer = self.renderer
        elif isinstance(renderer, int):
                renderer = self.renderers.index(renderer)
        else:
            return []

        if obj is None or isinstance(obj, int):
            if obj is None:
                acs = renderer.GetActors()
            elif obj >= len(self.renderers):
                colors.printc("~timesError in getMeshes: non existing renderer", obj, c=1)
                return []
            else:
                acs = self.renderers[obj].GetActors()

            actors = []
            acs.InitTraversal()
            for i in range(acs.GetNumberOfItems()):
                a = acs.GetNextItem()
                if a.GetPickable():
                    r = self.renderers.index(renderer)
                    if a == self.axes_instances[r]:
                        continue
                    actors.append(a)
            return actors

        elif isinstance(obj, vtk.vtkAssembly):
            cl = vtk.vtkPropCollection()
            obj.GetActors(cl)
            actors = []
            cl.InitTraversal()
            for i in range(obj.GetNumberOfPaths()):
                act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
                if act.GetPickable():
                    actors.append(act)
            return actors

        elif isinstance(obj, str):  # search the actor by the legend name
            actors = []
            for a in self.actors:
                if hasattr(a, "_legend") and obj in a._legend:
                    actors.append(a)
            return actors

        elif isinstance(obj, vtk.vtkActor):
            return [obj]

        if self.verbose:
            colors.printc("~lightning Warning in getMeshes: unexpected input type", obj, c=1)
        return []


    def moveCamera(self, camstart, camstop, fraction):
        """
        Takes as input two ``vtkCamera`` objects and returns a
        new ``vtkCamera`` that is at an intermediate position:

        fraction=0 -> camstart,  fraction=1 -> camstop.

        Press ``shift-C`` key in interactive mode to dump a python snipplet
        of parameters for the current camera view.
        """
        if isinstance(fraction, int):
            colors.printc("~lightning Warning in moveCamera(): fraction should not be an integer", c=1)
        if fraction > 1:
            colors.printc("~lightning Warning in moveCamera(): fraction is > 1", c=1)
        cam = vtk.vtkCamera()
        cam.DeepCopy(camstart)
        p1 = np.array(camstart.GetPosition())
        f1 = np.array(camstart.GetFocalPoint())
        v1 = np.array(camstart.GetViewUp())
        c1 = np.array(camstart.GetClippingRange())
        s1 = camstart.GetDistance()

        p2 = np.array(camstop.GetPosition())
        f2 = np.array(camstop.GetFocalPoint())
        v2 = np.array(camstop.GetViewUp())
        c2 = np.array(camstop.GetClippingRange())
        s2 = camstop.GetDistance()
        cam.SetPosition(p2 * fraction + p1 * (1 - fraction))
        cam.SetFocalPoint(f2 * fraction + f1 * (1 - fraction))
        cam.SetViewUp(v2 * fraction + v1 * (1 - fraction))
        cam.SetDistance(s2 * fraction + s1 * (1 - fraction))
        cam.SetClippingRange(c2 * fraction + c1 * (1 - fraction))
        self.camera = cam
        save_int = self.interactive
        self.show(resetcam=0, interactive=0)
        self.interactive = save_int
        return cam

    ##################################################################
    def addLight(self, pos, focalPoint=(0, 0, 0), deg=180, c='white',
                 intensity=0.4, removeOthers=False, showsource=False):
        """
        Generate a source of light placed at pos, directed to focal point.
        Returns a ``vtkLight`` object.

        :param focalPoint: focal point, if this is a ``vtkActor`` use its position.
        :type fp: vtkActor, list
        :param deg: aperture angle of the light source
        :param c: set light color
        :param float intensity: intensity between 0 and 1.
        :param bool removeOthers: remove all other lights in the scene
        :param bool showsource: if `True`, will show a representation
                                of the source of light as an extra Mesh

        .. hint:: |lights.py|_
        """
        return addons.addLight(pos, focalPoint, deg, c,
                               intensity, removeOthers, showsource)

    def addSlider2D(self, sliderfunc, xmin, xmax,
                    value=None, pos=4, title="", c=None, showValue=True):
        """Add a slider widget which can call an external custom function.

        :param sliderfunc: external function to be called by the widget
        :param float xmin:  lower value
        :param float xmax:  upper value
        :param float value: current value
        :param list pos:  position corner number: horizontal [1-4] or vertical [11-14]
                            it can also be specified by corners coordinates [(x1,y1), (x2,y2)]
        :param str title: title text
        :param bool showValue:  if true current value is shown

        |sliders| |sliders.py|_
        """
        return addons.addSlider2D(sliderfunc, xmin, xmax, value, pos, title, c, showValue)

    def addSlider3D(
        self,
        sliderfunc,
        pos1,
        pos2,
        xmin,
        xmax,
        value=None,
        s=0.03,
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
        :param str title: title text
        :param c: slider color
        :param float rotation: title rotation around slider axis
        :param bool showValue: if True current value is shown

        |sliders3d| |sliders3d.py|_
        """
        return addons.addSlider3D(
            sliderfunc, pos1, pos2, xmin, xmax, value, s, title, rotation, c, showValue
        )

    def addButton(
        self,
        fnc,
        states=("On", "Off"),
        c=("w", "w"),
        bc=("dg", "dr"),
        pos=(20, 40),
        size=24,
        font="arial",
        bold=False,
        italic=False,
        alpha=1,
        angle=0,
    ):
        """Add a button to the renderer window.

        :param list states: a list of possible states, e.g. ['On', 'Off']
        :param c:      a list of colors for each state
        :param bc:     a list of background colors for each state
        :param pos:    2D position in pixels from left-bottom corner
        :param size:   size of button font
        :param str font:   font type (arial, courier, times)
        :param bool bold:   bold face (False)
        :param bool italic: italic face (False)
        :param float alpha:  opacity level
        :param float angle:  anticlockwise rotation in degrees

        |buttons| |buttons.py|_
        """
        return addons.addButton(fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle)

    def addCutterTool(self, mesh):
        """Create handles to cut away parts of a mesh.

        |cutter| |cutter.py|_
        """
        return addons.addCutterTool(mesh)

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
              - 7,  draw a simple ruler at the bottom of the window
              - 8,  show the ``vtkCubeAxesActor`` object
              - 9,  show the bounding box outLine
              - 10, show three circles representing the maximum bounding box
              - 11, show a large grid on the x-y plane (use with zoom=8)
              - 12, show polar axes.

        Axis type-1 can be fully customized by passing a dictionary ``axes=dict()`` where:

            - `xtitle`,            ['x'], x-axis title text.
            - `ytitle`,            ['y'], y-axis title text.
            - `ztitle`,            ['z'], z-axis title text.
            - `numberOfDivisions`,   [4], number of divisions on the longest axis
            - `axesLineWidth`,       [1], width of the axes lines
            - `gridLineWidth`,       [1], width of the grid lines
            - `reorientShortTitle`, [True], titles shorter than 3 letters are placed horizontally
            - `originMarkerSize`, [0.01], draw a small cube on the axis where the origin is
            - `enableLastLabel`, [False], show last numeric label on axes
            - `titleDepth`,          [0], extrusion fractional depth of title text
            - `xyGrid`,           [True], show a gridded wall on plane xy
            - `yzGrid`,           [True], show a gridded wall on plane yz
            - `zxGrid`,          [False], show a gridded wall on plane zx
            - `zxGrid2`,         [False], show zx plane on opposite side of the bounding box
            - `xyPlaneColor`,   ['gray'], color of gridded plane
            - `xyGridColor`,    ['gray'], grid line color
            - `xyAlpha`,          [0.15], grid plane opacity
            - `showTicks`,        [True], show major ticks
            - `xTitlePosition`,   [0.32], title fractional positions along axis
            - `xTitleOffset`,     [0.05], title fractional offset distance from axis line
            - `xTitleJustify`, ["top-right"], title justification
            - `xTitleRotation`,      [0], add a rotation of the axis title
            - `xLineColor`,  [automatic], color of the x-axis
            - `xTitleColor`, [automatic], color of the axis title
            - `xTitleBackfaceColor`, [None],  color of axis title on its backface
            - `xTitleSize`,      [0.025], size of the axis title
            - `xHighlightZero`,   [True], draw a line highlighting zero position if in range
            - `xHighlightZeroColor`, [automatic], color of the line highlighting the zero position
            - `xTickRadius`,     [0.005], radius of the major ticks
            - `xTickThickness`, [0.0025], thickness of the major ticks along their axis
            - `xTickColor`,  [automatic], color of major ticks
            - `xMinorTicks`,         [1], number of minor ticks between two major ticks
            - `tipSize`,          [0.01], size of the arrow tip
            - `xLabelPrecision`,     [2], nr. of significative digits to be shown
            - `xLabelSize`,      [0.015], size of the numeric labels along axis
            - `xLabelOffset`,    [0.025], offset of numeric labels

            :Example:

                .. code-block:: python

                    from vtkplotter import Box, show
                    b = Box(pos=(0,0,0), length=80, width=90, height=70).alpha(0)

                    show(b, axes={ 'xtitle':'Some long variable [a.u.]',
                                   'numberOfDivisions':4,
                                   # ...
                                 }
                    )

        |customAxes| |customAxes.py|_
        """
        return addons.addGlobalAxes(axtype, c)

    def addLegend(self):
        return addons.addLegend()


    ##############################################################################
    def show(self, *actors, **options):
        """Render a list of actors.

        Allowed input objects are: ``filename``, ``vtkPolyData``, ``vtkActor``,
        ``vtkActor2D``, ``vtkImageActor``, ``vtkAssembly`` or ``vtkVolume``.

        If filename is given, its type is guessed based on its extension.
        Supported formats are:
        `vtu, vts, vtp, ply, obj, stl, 3ds, xml, neutral, gmsh, pcd, xyz, txt, byu,
        tif, slc, vti, mhd, png, jpg`.

        :param int at: number of the renderer to plot to, if more than one exists
        :param list shape: Number of sub-render windows inside of the main window.
            Specify two across with ``shape=(2, 1)`` and a two by two grid with ``shape=(2, 2)``.
            By default there is only one renderer.
            Can also accept a shape as string descriptor. E.g.

            - shape="3|1" means 3 plots on the left and 1 on the right,
            - shape="4/2" means 4 plots on top of 2 at bottom.

        :param int axes: set the type of axes to be shown

              - 0,  no axes,
              - 1,  draw three customizable gray grid walls
              - 2,  show cartesian axes from (0,0,0)
              - 3,  show positive range of cartesian axes from (0,0,0)
              - 4,  show a triad at bottom left
              - 5,  show a cube at bottom left
              - 6,  mark the corners of the bounding box
              - 7,  draw a simple ruler at the bottom of the window
              - 8,  show the ``vtkCubeAxesActor`` object,
              - 9,  show the bounding box outLine,
              - 10, show three circles representing the maximum bounding box
              - 11, show a large grid on the x-y plane (use with zoom=8)
              - 12, show polar axes.

        :param float azimuth/elevation/roll:  move camera accordingly
        :param str viewup:  either ['x', 'y', 'z'] or a vector to set vertical direction
        :param bool resetcam:  re-adjust camera position to fit objects
        :param dict camera: Camera parameters can further be specified with a dictionary assigned
           to the ``camera`` keyword (E.g. `show(camera={'pos':(1,2,3), 'thickness':1000,})`)

            - pos, `(list)`,  the position of the camera in world coordinates
            - focalPoint `(list)`, the focal point of the camera in world coordinates
            - viewup `(list)`, the view up direction for the camera
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
        :param int interactorStyle: set the type of interaction
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
            -10 = Terrain
            -11 = Unicam

        :param bool q:  force program to quit after `show()` command returns.
        """
        at = options.pop("at", None)
        axes = options.pop("axes", None)
        resetcam = options.pop("resetcam", True)
        zoom = options.pop("zoom", False)
        interactive = options.pop("interactive", None)
        viewup = options.pop("viewup", "")
        azimuth = options.pop("azimuth", 0)
        elevation = options.pop("elevation", 0)
        roll = options.pop("roll", 0)
        camera = options.pop("camera", None)
        interactorStyle = options.pop("interactorStyle", 0)
        rate = options.pop("rate", None)
        q = options.pop("q", False)

        if self.offscreen:
            interactive = False
            self.interactive = False

        def scan(wannabeacts):
            scannedacts = []
            if not utils.isSequence(wannabeacts):
                wannabeacts = [wannabeacts]

            for a in wannabeacts:  # scan content of list

                if a is None:
                    pass

                elif isinstance(a, vtk.vtkActor):
                    scannedacts.append(a)
                    if hasattr(a, 'trail') and a.trail and a.trail not in self.actors:
                        scannedacts.append(a.trail)
                    if hasattr(a, 'shadow') and a.shadow and a.shadow not in self.actors:
                        scannedacts.append(a.shadow)

                elif isinstance(a, vtk.vtkAssembly):
                    scannedacts.append(a)
                    if a.trail and a.trail not in self.actors:
                        scannedacts.append(a.trail)

                elif isinstance(a, vtk.vtkActor2D):
                    if isinstance(a, vtk.vtkCornerAnnotation):
                        for a2 in settings.collectable_actors:
                            if isinstance(a2, vtk.vtkCornerAnnotation):
                                if at in a2.renderedAt: # remove old message
                                    self.remove(a2)
                    scannedacts.append(a)

                elif a is Ellipsis:
                    scannedacts += settings.collectable_actors

                elif isinstance(a, vtk.vtkImageActor):
                    scannedacts.append(a)

                elif isinstance(a, Volume):
                    scannedacts.append(a)

                elif isinstance(a, vtk.vtkVolume):
                    scannedacts.append(Volume(a.GetMapper().GetInput()))

                elif isinstance(a, vtk.vtkImageData):
                    scannedacts.append(Volume(a))

                elif isinstance(a, vtk.vtkBillboardTextActor3D):
                    scannedacts.append(a)

                elif isinstance(a, str):  # assume a filepath was given
                    out = vtkio.load(a)
                    scannedacts.append(out)

                elif isinstance(a, vtk.vtkMultiBlockDataSet):
                    for i in range(a.GetNumberOfBlocks()):
                        b =  a.GetBlock(i)
                        if isinstance(b, vtk.vtkPolyData):
                            scannedacts.append(Mesh(b))
                        elif isinstance(b, vtk.vtkImageData):
                            scannedacts.append(Volume(b))

                elif "dolfin" in str(type(a)):  # assume a dolfin.Mesh object
                    from vtkplotter.dolfin import MeshActor
                    out = MeshActor(a)
                    scannedacts.append(out)

                elif "trimesh" in str(type(a)):
                    from vtkplotter.utils import trimesh2vtk
                    scannedacts.append(trimesh2vtk(a))

                else:
                    try:
                        scannedacts.append(Mesh(a))
                    except:
                        colors.printc("~!? Cannot understand input in show():", type(a), c=1)
            return scannedacts

        if len(actors) == 0:
            actors = None
        elif len(actors) == 1:
            actors = actors[0]
        else:
            actors = utils.flatten(actors)

        if actors is not None:
            self.actors = []
            actors2show = scan(actors)
            for a in actors2show:
                if a not in self.actors:
                    self.actors.append(a)
        else:
            actors2show = scan(self.actors)
            self.actors = list(actors2show)

        if axes is not None:
            self.axes = axes

        #########################################################################
        if (settings.notebookBackend
            and settings.notebookBackend != "panel"
            and settings.notebookBackend.lower() != "2d"):
            return backends.getNotebookBackend(actors2show, zoom, viewup)
        #########################################################################

        if not hasattr(self, 'window'):
            return None

        if interactive is not None:
            self.interactive = interactive

        if at is None and len(self.renderers) > 1:
            # in case of multiple renderers a call to show w/o specifying
            # at which renderer will just render the whole thing and return
            if self.interactor:
                if zoom:
                    self.camera.Zoom(zoom)
                self.interactor.Render()
                if self.interactive:
                    self.interactor.Start()
                return

        if at is None:
            at = 0

        if at < len(self.renderers):
            self.renderer = self.renderers[at]
        else:
            colors.printc("~times Error in show(): wrong renderer index", at, c=1)
            return

        if self.qtWidget is not None:
            self.qtWidget.GetRenderWindow().AddRenderer(self.renderer)

        if not self.camera:
            if isinstance(camera, vtk.vtkCamera):
                self.camera = camera
            else:
                self.camera = self.renderer.GetActiveCamera()

        self.camera.SetParallelProjection(settings.useParallelProjection)

        if self.sharecam:
            for r in self.renderers:
                r.SetActiveCamera(self.camera)

        if len(self.renderers) == 1:
            self.renderer.SetActiveCamera(self.camera)

        # rendering
        for ia in actors2show:  # add the actors that are not already in scene
            if ia:
                if isinstance(ia, vtk.vtkVolume):
                    self.renderer.AddVolume(ia)
                else:
                    self.renderer.AddActor(ia)

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

                if hasattr(ia, 'GetTextProperty'):
                    #fix gray color of corner annotations
                    cacol = np.array(ia.GetTextProperty().GetColor())
                    if np.linalg.norm(cacol-(.5,.5,.5))/3 < 0.05:
                        c = (0.9, 0.9, 0.9)
                        if np.sum(self.renderer.GetBackground()) > 1.5:
                            c = (0.1, 0.1, 0.1)
                        ia.GetTextProperty().SetColor(c)

                if hasattr(ia, 'flagText') and self.interactor and not self.offscreen:
                    #check balloons
                    if ia.flagText:
                        if not self.flagWidget: # Create widget on the fly
                            self._flagRep = vtk.vtkBalloonRepresentation()
                            self._flagRep.SetBalloonLayoutToImageRight()
                            breppr = self._flagRep.GetTextProperty()
                            breppr.SetFontFamilyAsString(settings.flagFont)
                            breppr.SetFontSize(settings.flagFontSize)
                            breppr.SetBold(settings.flagBold)
                            breppr.SetItalic(settings.flagItalic)
                            breppr.SetColor(colors.getColor(settings.flagColor))
                            breppr.SetBackgroundColor(colors.getColor(settings.flagBackgroundColor))
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
        for ia in self.getMeshes(at) + self.getVolumes(at):
            if ia not in actors2show:
                self.renderer.RemoveActor(ia)
                if hasattr(ia, 'scalarbar') and ia.scalarbar:
                    if isinstance(ia.scalarbar, vtk.vtkActor):
                        self.renderer.RemoveActor(ia.scalarbar)
                    elif isinstance(ia.scalarbar, Assembly):
                        for a in ia.scalarbar.getMeshes():
                            self.renderer.RemoveActor(a)
                if hasattr(ia, 'renderedAt'):
                    ia.renderedAt.discard(at)


        if self.axes is not None:
            addons.addGlobalAxes()

        #########################################################################
        if settings.notebookBackend == "panel":
            return backends.getNotebookBackend(0, 0, 0)
        #########################################################################

        addons.addLegend()

        if resetcam: #or self.initializedIren == False:
            self.renderer.ResetCamera()

        if settings.showRendererFrame and len(self.renderers) > 1:
            addons.addRendererFrame(c=settings.rendererFrameColor)

        if not self.initializedIren and self.interactor:
            self.initializedIren = True
            self.interactor.Initialize()
            self.interactor.RemoveObservers("CharEvent")

            if self.verbose and self.interactive:
                if not settings.notebookBackend:
                    docs.onelinetip()

        if self.flagWidget:
            self.flagWidget.EnabledOn()

        self.initializedPlotter = True

        if zoom:
            self.camera.Zoom(zoom)
        if azimuth:
            self.camera.Azimuth(azimuth)
        if elevation:
            self.camera.Elevation(elevation)
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
            elif isinstance(viewup, str) and viewup.lower() == "2d":
                interactorStyle = 7

        if camera is not None:
            cm_pos = camera.pop("pos", None)
            cm_focalPoint = camera.pop("focalPoint", None)
            cm_viewup = camera.pop("viewup", None)
            cm_distance = camera.pop("distance", None)
            cm_clippingRange = camera.pop("clippingRange", None)
            cm_parallelScale = camera.pop("parallelScale", None)
            cm_thickness = camera.pop("thickness", None)
            cm_viewAngle = camera.pop("viewAngle", None)
            if cm_pos is not None: self.camera.SetPosition(cm_pos)
            if cm_focalPoint is not None: self.camera.SetFocalPoint(cm_focalPoint)
            if cm_viewup is not None: self.camera.SetViewUp(cm_viewup)
            if cm_distance is not None: self.camera.SetDistance(cm_distance)
            if cm_clippingRange is not None: self.camera.SetClippingRange(cm_clippingRange)
            if cm_parallelScale is not None: self.camera.SetParallelScale(cm_parallelScale)
            if cm_thickness is not None: self.camera.SetThickness(cm_thickness)
            if cm_viewAngle is not None: self.camera.SetViewAngle(cm_viewAngle)

        if resetcam:
            self.renderer.ResetCameraClippingRange()

        self.window.Render() ############################# <----


        #########################################################################
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
            pass  # do nothing
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

        if hasattr(self, 'interactor') and self.interactor and self.interactive:
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
            if self.verbose:
                print("q flag set to True.  Exit python session.")
            sys.exit(0)

        return self


    def showInset(self, *actors, **options):
        """Add a draggable inset space into a renderer.

        :param pos: icon position in the range [1-4] indicating one of the 4 corners,
                    or it can be a tuple (x,y) as a fraction of the renderer size.
        :param float size: size of the square inset.
        :param bool draggable: if True the subrenderer space can be dragged around.

        |inset| |inset.py|_
        """
        pos = options.pop("pos", None)
        size = options.pop("size", 0.1)
        c = options.pop("c", 'r')
        draggable = options.pop("draggable", True)

        if not self.renderer:
            colors.printc("~lightningWarning: Use showInset() after first rendering the scene.",
                          c=3)
            save_int = self.interactive
            self.show(interactive=0)
            self.interactive = save_int
        widget = vtk.vtkOrientationMarkerWidget()
        r,g,b = colors.getColor(c)
        widget.SetOutlineColor(r,g,b)
        if len(actors)==1:
            widget.SetOrientationMarker(actors[0])
        else:
            widget.SetOrientationMarker(Assembly(utils.flatten(actors)))
        widget.SetInteractor(self.interactor)
        if utils.isSequence(pos):
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
        self.widgets.append(widget)
        for a in actors:
            if a in self.actors:
                self.actors.remove(a)
        return widget

    def clear(self, actors=()):
        """Delete specified list of actors, by default delete all."""
        if not utils.isSequence(actors):
            actors = [actors]
        if len(actors):
            for a in actors:
                self.remove(a)
        elif self.renderer:
            for a in settings.collectable_actors:
                self.remove(a)
            settings.collectable_actors = []
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
                self.interactor.TerminateApp()
                del self.window
                del self.interactor
        return self

    def close(self):
        self.closeWindow()
        self.actors = []
        settings.collectable_actors = []
        return None


    #######################################################################
    def _mouseleft(self, iren, event):

        x, y = iren.GetEventPosition()
        #print('_mouseleft mouse at', x, y)

        renderer = iren.FindPokedRenderer(x, y)
        self.renderer = renderer

        picker = vtk.vtkPropPicker()
        picker.PickProp(x, y, renderer)
        clickedActor = picker.GetActor()

        # check if any button objects are clicked
        clickedActor2D = picker.GetActor2D()
        if clickedActor2D:
            for bt in self.buttons:
                if clickedActor2D == bt.actor:
                    bt.function()
                    break

        if not clickedActor:
            clickedActor = picker.GetAssembly()

        self.picked3d = picker.GetPickPosition()

        self.justremoved = None

        if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
            return

        self.clickedActor = clickedActor
        if hasattr(clickedActor, 'picked3d'):
            clickedActor.picked3d = picker.GetPickPosition()

        if self.mouseLeftClickFunction:
            self.mouseLeftClickFunction(clickedActor)


    def _mouseright(self, iren, event):

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
        self.picked3d = picker.GetPickPosition()

        if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
            return
        self.clickedActor = clickedActor

        if self.mouseRightClickFunction:
            self.mouseRightClickFunction(clickedActor)


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
        self.picked3d = picker.GetPickPosition()

        if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
            return
        self.clickedActor = clickedActor

        if self.mouseMiddleClickFunction:
            self.mouseMiddleClickFunction(self.clickedActor)


    def _keypress(self, iren, event):
        # qt creates and passes a vtkGenericRenderWindowInteractor

        key = iren.GetKeySym()
        #print('Pressed key:', key, [vp])

        if key in ["q", "Q", "space", "Return"]:
            iren.ExitCallback()
            return

        elif key == "Escape":
            sys.stdout.flush()
            settings.plotter_instance.closeWindow()

        elif key in ["F1", "Pause"]:
            sys.stdout.flush()
            colors.printc('\n[F1] pressed. Execution aborted. Exiting python now.', c=1)
            settings.plotter_instance.close()
            sys.exit(0)

        elif key == "m":
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

        elif key == "comma":
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

        elif key == "period":
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

        elif key == "slash":
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

        #############################################################
        ### now intercept custom observer ###########################
        #############################################################
        if self.keyPressFunction:
            if key not in ["Shift_L", "Control_L", "Super_L", "Alt_L"]:
                if key not in ["Shift_R", "Control_R", "Super_R", "Alt_R"]:
                    self.verbose = False
                    self.keyPressFunction(key)
                    return

        if key == "h":
            from vtkplotter.docs import tips

            tips()
            return

        if key == "a":
            iren.ExitCallback()
            cur = iren.GetInteractorStyle()
            if isinstance(cur, vtk.vtkInteractorStyleTrackballCamera):
                print("\nInteractor style changed to TrackballActor")
                print("  you can now move and rotate individual meshes:")
                print("  press X twice to save the repositioned mesh,")
                print("  press 'a' to go back to normal style.")
                iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
            else:
                iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            iren.Start()
            return

        if key == "j":
            iren.ExitCallback()
            cur = iren.GetInteractorStyle()
            if isinstance(cur, vtk.vtkInteractorStyleJoystickCamera):
                iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            else:
                print("\nInteractor style changed to Joystick,", end="")
                print(" press j to go back to normal.")
                iren.SetInteractorStyle(vtk.vtkInteractorStyleJoystickCamera())
            iren.Start()
            return

        if key == "S":
            vtkio.screenshot("screenshot.png")
            colors.printc("~camera Saved rendering window as screenshot.png", c="blue")
            return

        if key == "C":
            cam = self.renderer.GetActiveCamera()
            print('\n### Example code to position this vtkCamera:')
            print('vp = vtkplotter.Plotter()\n...')
            print('vp.camera.SetPosition(',   [round(e, 3) for e in cam.GetPosition()],  ')')
            print('vp.camera.SetFocalPoint(', [round(e, 3) for e in cam.GetFocalPoint()], ')')
            print('vp.camera.SetViewUp(',     [round(e, 3) for e in cam.GetViewUp()], ')')
            print('vp.camera.SetDistance(',   round(cam.GetDistance(), 3), ')')
            print('vp.camera.SetClippingRange(',
                                    [round(e, 3) for e in cam.GetClippingRange()], ')')
            return

        if key == "s":
            if self.clickedActor and self.clickedActor in self.getMeshes():
                self.clickedActor.GetProperty().SetRepresentationToSurface()
            else:
                for a in self.getMeshes():
                    if a and a.GetPickable():
                        a.GetProperty().SetRepresentationToSurface()

        elif key == "V":
            if not (self.verbose):
                self._tips()
            self.verbose = not (self.verbose)
            print("Verbose: ", self.verbose)

        elif key == "1":
            self.icol += 1
            if self.clickedActor and hasattr(self.clickedActor, "GetProperty"):
                self.clickedActor.GetMapper().ScalarVisibilityOff()
                self.clickedActor.GetProperty().SetColor(colors.colors1[(self.icol) % 10])
            else:
                for i, ia in enumerate(self.getMeshes()):
                    if not ia.GetPickable():
                        continue
                    ia.GetProperty().SetColor(colors.colors1[(i + self.icol) % 10])
                    ia.GetMapper().ScalarVisibilityOff()
            addons.addLegend()

        elif key == "2":
            self.icol += 1
            if self.clickedActor and hasattr(self.clickedActor, "GetProperty"):
                self.clickedActor.GetMapper().ScalarVisibilityOff()
                self.clickedActor.GetProperty().SetColor(colors.colors2[(self.icol) % 10])
            else:
                for i, ia in enumerate(self.getMeshes()):
                    if not ia.GetPickable():
                        continue
                    ia.GetProperty().SetColor(colors.colors2[(i + self.icol) % 10])
                    ia.GetMapper().ScalarVisibilityOff()
            addons.addLegend()

        elif key == "3":
            c = colors.getColor("gold")
            acs = self.getMeshes()
            if len(acs) == 0: return
            alpha = 1.0 / len(acs)
            for ia in acs:
                if not ia.GetPickable():
                    continue
                ia.GetProperty().SetColor(c)
                ia.GetProperty().SetOpacity(alpha)
                ia.GetMapper().ScalarVisibilityOff()
            addons.addLegend()

        elif key == "4":
            for ia in self.getMeshes():
                if not ia.GetPickable():
                    continue
                if isinstance(ia, Mesh):
                    iascals = ia.getPointArray()
                    if len(iascals):
                        stype, sname = iascals[ia._scals_idx]
                        if sname and "Normals" not in sname.lower(): # exclude normals
                            ia.getPointArray( ia._scals_idx )
                            colors.printc("..active scalars set to:", sname,
                                          "\ttype:", stype, c='g', bold=0)
                        ia._scals_idx += 1
                        if ia._scals_idx >= len(iascals):
                            ia._scals_idx = 0
            addons.addLegend()

        elif key == "5":
            bgc = np.array(self.renderer.GetBackground()).sum() / 3
            if bgc <= 0:
                bgc = 0.223
            elif 0 < bgc < 1:
                bgc = 1
            else:
                bgc = 0
            self.renderer.SetBackground(bgc, bgc, bgc)

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
                        self.renderer.RemoveActor(self.axes_instances[clickedr])
                    self.axes_instances[clickedr] = None
                addons.addGlobalAxes(axtype=asso[key], c=None)
                self.interactor.Render()

        if key == "O":
            settings.plotter_instance.renderer.RemoveLight(self.extralight)
            self.extralight = None

        elif key == "o":
            vbb, sizes, _, _ = addons.computeVisibleBounds()
            cm = utils.vector((vbb[0]+vbb[1])/2, (vbb[2]+vbb[3])/2, (vbb[4]+vbb[5])/2)
            if not self.extralight:
                vup = self.renderer.GetActiveCamera().GetViewUp()
                pos = cm + utils.vector(vup)*utils.mag(sizes)
                self.extralight = addons.addLight(pos, focalPoint=cm)
                print("Press again o to rotate light source, or O to remove it.")
            else:
                cpos = utils.vector(self.extralight.GetPosition())
                x, y, z = self.extralight.GetPosition() - cm
                r,th,ph = utils.cart2spher(x,y,z)
                th += 0.2
                if th>np.pi: th=np.random.random()*np.pi/2
                ph += 0.3
                cpos = utils.spher2cart(r, th,ph) + cm
                self.extralight.SetPosition(cpos)

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
                    'glossy')
            for ia in acts:
                if ia.GetPickable():
                    try:
                        lnr = (ia._ligthingnr+1)%5
                        ia.lighting(shds[lnr])
                        ia._ligthingnr = lnr
                        colors.printc('-> lighting set to:', shds[lnr], c='g', bold=0)
                    except AttributeError:
                        pass

        elif key == "K": # shading
            if self.clickedActor in self.getMeshes():
                acts = [self.clickedActor]
            else:
                acts = self.getMeshes()
            for ia in acts:
                if ia.GetPickable():
                    ia.computeNormals()
                    intrp = (ia.GetProperty().GetInterpolation()+1)%3
                    ia.GetProperty().SetInterpolation(intrp)
                    colors.printc('->  shading set to:',
                                  ia.GetProperty().GetInterpolationAsString(),
                                  c='g', bold=0)

        elif key == "n":  # show normals to an actor
            from vtkplotter.analysis import normalLines

            if self.clickedActor in self.getMeshes():
                if self.clickedActor.GetPickable():
                    self.renderer.AddActor(normalLines(self.clickedActor))
                    iren.Render()
            else:
                print("Click an actor and press n to add normals.")


        elif key == "x":
            if self.justremoved is None:
                if self.clickedActor in self.getMeshes() \
                  or isinstance(self.clickedActor, vtk.vtkAssembly):
                    self.justremoved = self.clickedActor
                    self.renderer.RemoveActor(self.clickedActor)
                if hasattr(self.clickedActor, '_legend') and self.clickedActor._legend:
                    print('...removing actor: ' +
                          str(self.clickedActor._legend)+', press x to put it back')
                else:
                    print("Click an actor and press x to toggle it.")
            else:
                self.renderer.AddActor(self.justremoved)
                self.renderer.Render()
                self.justremoved = None
            addons.addLegend()

        elif key == "X":
            if self.clickedActor:
                if not self.cutterWidget:
                    addons.addCutterTool(self.clickedActor)
                else:
                    fname = "clipped.vtk"
                    confilter = vtk.vtkPolyDataConnectivityFilter()
                    if isinstance(self.clickedActor, vtk.vtkActor):
                        confilter.SetInputData(self.clickedActor.GetMapper().GetInput())
                    elif isinstance(self.clickedActor, vtk.vtkAssembly):
                        act = self.clickedActor.getMeshes()[0]
                        confilter.SetInputData(act.GetMapper().GetInput())
                    else:
                        confilter.SetInputData(self.clickedActor.polydata(True))
                    confilter.SetExtractionModeToLargestRegion()
                    confilter.Update()
                    cpd = vtk.vtkCleanPolyData()
                    cpd.SetInputData(confilter.GetOutput())
                    cpd.Update()
                    w = vtk.vtkPolyDataWriter()
                    w.SetInputData(cpd.GetOutput())
                    w.SetFileName(fname)
                    w.Write()
                    colors.printc("~save Saved file:", fname, c="m")
                    self.cutterWidget.Off()
                    self.cutterWidget = None
            else:
                for a in self.actors:
                    if isinstance(a, vtk.vtkVolume):
                        addons.addCutterTool(a)
                        return

                colors.printc("Click object and press X to open the cutter box widget.",
                              c=4)

        elif key == "E":
            colors.printc("~camera Exporting rendering window to scene.npy..",
                          c="blue", end="")
            vtkio.exportWindow('scene.npy')
            colors.printc(" ..done. Try:\n> vtkplotter scene.npy  #(still experimental)",
                          c="blue")

        elif key == "i":  # print info
            if self.clickedActor:
                utils.printInfo(self.clickedActor)
            else:
                utils.printInfo(self)

        if iren:
            iren.Render()

