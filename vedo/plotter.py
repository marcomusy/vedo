from __future__ import division, print_function
import time
import sys
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

import vedo
from vedo.colors import printc, getColor
import vedo.utils as utils
import vedo.docs as docs
import vedo.settings as settings
import vedo.addons as addons
import vedo.backends as backends

__doc__ = (
    """
Defines main class ``Plotter`` to manage actors and 3D rendering.
"""
    + docs._defs
)

__all__ = ["show", "clear", "ion", "ioff",
           "Plotter", "closeWindow", "closePlotter", "interactive"]

########################################################################
def show(*actors, **options):
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
            - 11, show a large grid on the x-y plane (use with zoom=8)
            - 12, show polar axes
            - 13, draw a simple ruler at the bottom of the window

        Axis type-1 can be fully customized by passing a dictionary ``axes=dict()`` where:

            - `xtitle`,                ['x'], x-axis title text
            - `xrange`,               [None], x-axis range in format (xmin, ymin), default is automatic.
            - `numberOfDivisions`,    [None], approximate number of divisions on the longest axis
            - `axesLineWidth`,           [1], width of the axes lines
            - `gridLineWidth`,           [1], width of the grid lines
            - `reorientShortTitle`,   [True], titles shorter than 2 letter are placed horizontally
            - `titleDepth`,              [0], extrusion fractional depth of title text
            - `xyGrid`,               [True], show a gridded wall on plane xy
            - `yzGrid`,               [True], show a gridded wall on plane yz
            - `zxGrid`,               [True], show a gridded wall on plane zx
            - `zxGrid2`,             [False], show zx plane on opposite side of the bounding box
            - `xyGridTransparent`    [False], make grid plane completely transparent
            - `xyGrid2Transparent`   [False], make grid plane completely transparent on opposite side box
            - `xyPlaneColor`,       ['gray'], color of the plane
            - `xyGridColor`,        ['gray'], grid line color
            - `xyAlpha`,              [0.15], grid plane opacity
            - `xyFrameLine`,          [None], add a frame for the plane
            - `showTicks`,            [True], show major ticks
            - `digits`,               [None], use this number of significant digits in scientific notation
            - `titleFont`,              [''], font for axes titles
            - `labelFont`,              [''], font for numeric labels
            - `textScale`,             [1.0], global scaling factor for text elements (titles, labels)
            - `xTitlePosition`,       [0.32], title fractional positions along axis
            - `xTitleOffset`,         [0.05], title fractional offset distance from axis line
            - `xTitleJustify`, ["top-right"], title justification
            - `xTitleRotation`,          [0], add a rotation of the axis title
            - `xTitleBox`,           [False], add a box around title text
            - `xLineColor`,      [automatic], color of the x-axis
            - `xTitleColor`,     [automatic], color of the axis title
            - `xTitleBackfaceColor`,  [None],  color of axis title on its backface
            - `xTitleSize`,          [0.025], size of the axis title
            - 'xTitleItalic',            [0], a bool or float to make the font italic
            - `xHighlightZero`,       [True], draw a line highlighting zero position if in range
            - `xHighlightZeroColor`, [autom], color of the line highlighting the zero position
            - `xTickLength`,         [0.005], radius of the major ticks
            - `xTickThickness`,     [0.0025], thickness of the major ticks along their axis
            - `xMinorTicks`,             [1], number of minor ticks between two major ticks
            - `xValuesAndLabels`          [], assign custom tick positions and labels [(pos1, label1), ...]
            - `xLabelColor`,     [automatic], color of numeric labels and ticks
            - `xLabelPrecision`,         [2], nr. of significative digits to be shown
            - `xLabelSize`,          [0.015], size of the numeric labels along axis
            - 'xLabelRotation',          [0], rotate clockwise [1] or anticlockwise [-1] by 90 degrees
            - 'xFlipText',           [False], flip axis title and numeric labels orientation
            - `xLabelOffset`,        [0.025], offset of numeric labels
            - `tipSize`,              [0.01], size of the arrow tip
            - `limitRatio`,           [0.04], below this ratio don't plot small axis

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

    :param bool new: if set to `True`, a call to ``show`` will instantiate
        a new ``Plotter`` object (a new window) instead of reusing the first created.

        See e.g.: |readVolumeAsIsoSurface.py|_

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
    at = options.pop("at", None)
    shape = options.pop("shape", (1, 1))
    N = options.pop("N", None)
    pos = options.pop("pos", (0, 0))
    size = options.pop("size", "auto")
    screensize = options.pop("screensize", "auto")
    title = options.pop("title", "")
    bg = options.pop("bg", "white")
    bg2 = options.pop("bg2", None)
    axes = options.pop("axes", settings.defaultAxesType)
    verbose = options.pop("verbose", False)
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
    q = options.pop("q", False)

    newPlotter = options.pop("new", False)
    newPlotter_old = options.pop("newPlotter", 'dont')
    if newPlotter_old != 'dont':
        printc("\nPlease use keyword new in show() instead of newPlotter\n", c='r')
        newPlotter = newPlotter_old

    if len(options):
        for op in options:
            printc("Warning: unknown keyword in show():", op, c='y')

    if len(actors) == 0:
        actors = None
    elif len(actors) == 1:
        actors = actors[0]
    else:
        actors = utils.flatten(actors)

    if actors is Ellipsis:
        actors = settings.collectable_actors

    if settings.plotter_instance and not newPlotter:
        plt = settings.plotter_instance
    else:
        if utils.isSequence(at):
            if not utils.isSequence(actors):
                printc("show() Error: input must be a list.", c='r')
                raise RuntimeError()
            if len(at) != len(actors):
                printc("show() Error: lists 'input' and 'at', must have equal lengths.", c='r')
                raise RuntimeError()
            if len(at) > 1 and (shape == (1, 1) and N is None):
                N = max(at) + 1
        elif at is None and (N or shape != (1, 1)):
            if not utils.isSequence(actors):
                printc('show() Error: N or shape is set, but input is not a sequence.', c='r')
                printc('              you may need to specify e.g. at=0', c='r')
                raise RuntimeError()
            at = range(len(actors))

        plt = Plotter(
                    shape=shape,
                    N=N,
                    pos=pos,
                    size=size,
                    screensize=screensize,
                    title=title,
                    axes=axes,
                    sharecam=sharecam,
                    verbose=verbose,
                    interactive=interactive,
                    offscreen=offscreen,
                    bg=bg,
                    bg2=bg2,
        )

    # use _plt_to_return because plt.show() can return a k3d/panel plot
    if utils.isSequence(at):
        for i, a in enumerate(actors):
            _plt_to_return = plt.show(
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
                bg=bg,
                bg2=bg2,
                axes=axes,
                q=q,
            )
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
            bg=bg,
            bg2=bg2,
            axes=axes,
            q=q,
        )

    return _plt_to_return


def interactive():
    """Start the rendering window interaction mode."""
    if settings.plotter_instance:
        if hasattr(settings.plotter_instance, 'interactor'):
            if settings.plotter_instance.interactor:
                settings.plotter_instance.interactor.Start()
    return settings.plotter_instance

def ion():
    """Set interactive mode ON.
    When calling ``show()`` python script exectution will stop and control
    will stay on the graphic window allowing mouse/keyboard interaction."""
    if settings.plotter_instance:
        settings.plotter_instance.interactive = True
    return settings.plotter_instance

def ioff():
    """Set interactive mode OFF.
    When calling ``show()`` image will be rendered but python script execution
    will continue, the graphic window will be not responsive to interaction."""
    if settings.plotter_instance:
        settings.plotter_instance.interactive = False
    return settings.plotter_instance


def clear(actor=None):
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
        from vedo.settings import plotter_instance
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
        settings.plotter_instance.close()
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
      - 7,  draw a 3D ruler at each side of the cartesian axes
      - 8,  show the ``vtkCubeAxesActor`` object
      - 9,  show the bounding box outLine,
      - 10, show three circles representing the maximum bounding box,
      - 11, show a large grid on the x-y plane (use with zoom=8)
      - 12, show polar axes.
      - 13, draw a simple ruler at the bottom of the window

    Axis type-1 can be fully customized by passing a dictionary ``axes=dict()`` where:

        - `xtitle`,            ['x'], x-axis title text.
        - `xrange`,           [None], x-axis range in format (xmin, ymin), default is automatic.
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
      See Also: example qt_windows1.py and qt_windows2.py

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
        bg="white",
        bg2=None,
        axes=settings.defaultAxesType,
        sharecam=True,
        verbose=False,
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

        if interactive is None:
            if N==1:
                interactive = True
            elif N or shape != (1, 1):
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
        self.resetcam= True
        self.allowInteraction = None

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
        if notebookBackend and notebookBackend != "panel" and notebookBackend != "2d":
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
                printc("Warning: having set N, shape is ignored.", c='r')
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
                r.SetBackground(getColor(self.backgrcol))
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
                arenderer.SetUseFXAA(settings.useFXAA)
                if settings.useFXAA:
                    self.window.SetMultiSamples(settings.multiSamples)
                arenderer.SetUseDepthPeeling(settings.useDepthPeeling)
                arenderer.SetViewport(x0, y0, x1, y1)
                arenderer.SetBackground(getColor(bg_))
                if bg2_:
                    arenderer.GradientBackgroundOn()
                    arenderer.SetBackground2(getColor(bg2_))

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

            ############################
            if sum(shape) > 3:
                settings.legendSize *= 2

            image_actor=None
            bgname = str(self.backgrcol).lower()
            if ".jpg" in bgname or ".jpeg" in bgname or ".png" in bgname:
                self.window.SetNumberOfLayers(2)
                self.backgroundRenderer = vtk.vtkRenderer()
                self.backgroundRenderer.SetLayer(0)
                self.backgroundRenderer.InteractiveOff()
                self.backgroundRenderer.SetBackground(getColor(bg2))
                image_actor = vedo.Picture(self.backgrcol)
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

                    arenderer.SetBackground(getColor(self.backgrcol))
                    if bg2:
                        arenderer.GradientBackgroundOn()
                        arenderer.SetBackground2(getColor(bg2))

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

        self.window.SetPosition(pos)

        if not title:
            title = " vedo " + vedo.__version__

        if self.qtWidget is not None:
            self.interactor = self.qtWidget.GetRenderWindow().GetInteractor()
            self.window = self.qtWidget.GetRenderWindow() # overwrite
            self.window.SetWindowName(title)
            ########################
            return
            ########################

        self.window.SetWindowName(title)

        for r in self.renderers:
            self.window.AddRenderer(r)

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

    def __iadd__(self, actors):
        self.add(actors, render=False)
        return self

    def __isub__(self, actors):
        self.remove(actors)
        return self

    def add(self, actors, render=True, at=None):
        """Append input object to the internal list of actors to be shown.

        :param bool render: render the scene after adding the object
        :param int at: add the object at the specified renderer

        :return: returns input actor for possible concatenation.
        """
        if at is not None:
            ren = self.renderers[at]
        else:
            ren = self.renderer

        if utils.isSequence(actors):
            for a in actors:
                if a not in self.actors:
                    self.actors.append(a)
                    if render and ren:
                        ren.AddActor(a)
            if render and self.interactor:
                if self.resetcam:
                    self.renderer.ResetCamera()
                self.interactor.Render()
            return None
        else:
            self.actors.append(actors)
            if render and ren:
                ren.AddActor(actors)
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

    def render(self):
        """Render the scene."""
        self.interactor.Render()
        return self

    def resetCamera(self):
        """Reset the camera position and zooming."""
        self.interactor.ResetCamera()
        return self

    def backgroundColor(self, c1=None, c2=None, at=None):
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
                r.SetBackground(getColor(c1))
            if c2 is not None:
                r.GradientBackgroundOn()
                r.SetBackground2(getColor(c2))
            else:
                r.GradientBackgroundOff()
        return self

    ####################################################
    def load(self, filename, unpack=True, force=False):
        """
        Load Mesh and Volume objects from file.
        The output will depend on the file extension. See examples below.

        :param bool unpack: only for multiblock data,
            if True returns a flat list of objects.
        :param bool force: when downloading a file ignore any previous
            cached downloads and force a new one.

        :Example:

            .. code-block:: python

                from vedo import datadir, load, show

                # Return a list of 2 Mesh
                g = load([datadir+'250.vtk', datadir+'290.vtk'])
                show(g)

                # Return a list of meshes by reading all files in a directory
                # (if directory contains DICOM files then a Volume is returned)
                g = load('mydicomdir/')
                show(g)

                # Return a Volume. Color/Opacity transfer function can be specified too.
                g = load(datadir+'embryo.slc')
                g.c(['y','lb','w']).alpha((0.0, 0.4, 0.9, 1)).show()
        """
        acts = vedo.io.load(filename, unpack, force)
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
                printc("Error in getVolumes(): non existing renderer", obj, c='r')
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
                printc("Error in getMeshes(): non existing renderer", obj, c='r')
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
            printc("Warning in getMeshes(): unexpected input type", obj, c='r')
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
            printc("Warning in moveCamera(): fraction should not be an integer", c='r')
        if fraction > 1:
            printc("Warning in moveCamera(): fraction is > 1", c='r')
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
                    value=None, pos=4, title="", font='arial', titleSize=1, c=None,
                    showValue=True):
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
        :param bool showValue:  if true current value is shown

        |sliders1| |sliders1.py|_ |sliders2.py|_
        """
        return addons.addSlider2D(sliderfunc, xmin, xmax, value,
                                  pos, title, font, titleSize, c, showValue)

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
            - 7,  draw a 3D ruler at each side of the cartesian axes
            - 8,  show the ``vtkCubeAxesActor`` object
            - 9,  show the bounding box outLine
            - 10, show three circles representing the maximum bounding box
            - 11, show a large grid on the x-y plane
            - 12, show polar axes
            - 13, draw a simple ruler at the bottom of the window

        Axis type-1 can be fully customized by passing a dictionary ``axes=dict()`` where:

            - `xtitle`,                ['x'], x-axis title text
            - `xrange`,               [None], x-axis range in format (xmin, ymin), default is automatic.
            - `numberOfDivisions`,    [None], approximate number of divisions on the longest axis
            - `axesLineWidth`,           [1], width of the axes lines
            - `gridLineWidth`,           [1], width of the grid lines
            - `reorientShortTitle`,   [True], titles shorter than 2 letter are placed horizontally
            - `originMarkerSize`,     [0.01], draw a small cube on the axis where the origin is
            - `titleDepth`,              [0], extrusion fractional depth of title text
            - `xyGrid`,               [True], show a gridded wall on plane xy
            - `yzGrid`,               [True], show a gridded wall on plane yz
            - `zxGrid`,               [True], show a gridded wall on plane zx
            - `zxGrid2`,             [False], show zx plane on opposite side of the bounding box
            - `xyGridTransparent`    [False], make grid plane completely transparent
            - `xyGrid2Transparent`   [False], make grid plane completely transparent on opposite side box
            - `xyPlaneColor`,       ['gray'], color of the plane
            - `xyGridColor`,        ['gray'], grid line color
            - `xyAlpha`,              [0.15], grid plane opacity
            - `xyFrameLine`,          [None], add a frame for the plane
            - `showTicks`,            [True], show major ticks
            - `xTitlePosition`,       [0.32], title fractional positions along axis
            - `xTitleOffset`,         [0.05], title fractional offset distance from axis line
            - `xTitleJustify`, ["top-right"], title justification
            - `xTitleRotation`,          [0], add a rotation of the axis title
            - `xLineColor`,      [automatic], color of the x-axis
            - `xTitleColor`,     [automatic], color of the axis title
            - `xTitleBackfaceColor`,  [None],  color of axis title on its backface
            - `xTitleSize`,          [0.025], size of the axis title
            - 'xTitleItalic',            [0], a bool or float to make the font italic
            - `xHighlightZero`,       [True], draw a line highlighting zero position if in range
            - `xHighlightZeroColor`, [autom], color of the line highlighting the zero position
            - `xTickLength`,         [0.005], radius of the major ticks
            - `xTickThickness`,     [0.0025], thickness of the major ticks along their axis
            - `xTickColor`,      [automatic], color of major ticks
            - `xMinorTicks`,             [1], number of minor ticks between two major ticks
            - `xPositionsAndLabels`       [], assign custom tick positions and labels [(pos1, label1), ...]
            - `xLabelPrecision`,         [2], nr. of significative digits to be shown
            - `xLabelSize`,          [0.015], size of the numeric labels along axis
            - `xLabelOffset`,        [0.025], offset of numeric labels
            - 'xFlipText'.           [False], flip axis title and numeric labels orientation
            - `tipSize`,              [0.01], size of the arrow tip
            - `limitRatio`,           [0.04], below this ratio don't plot small axis

            :Example:

                .. code-block:: python

                    from vedo import Box, show
                    b = Box(pos=(0,0,0), length=80, width=90, height=70).alpha(0)

                    show(b, axes={ 'xtitle':'Some long variable [a.u.]',
                                   'numberOfDivisions':4,
                                   # ...
                                 }
                    )

        |customAxes| |customAxes.py|_
        """
        addons.addGlobalAxes(axtype, c)
        return self

    def addLegend(self):
        return addons.addLegend()

    def addCallback(self, eventName, func):
        """Add a function to be executed while show() is active"""
        if self.interactor:
            self.interactor.AddObserver(eventName, func)
        return self

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
        axes = options.pop("axes", settings.defaultAxesType)
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
        bg_ = options.pop("bg", None)
        bg2_ = options.pop("bg2", None)
        axes_ = options.pop("axes", None)
        q = options.pop("q", False)

        if not settings.notebookBackend:
            if bg_ is not None:
                self.backgrcol = getColor(bg_)
                self.renderer.SetBackground(self.backgrcol)
            if bg2_ is not None:
                self.renderer.GradientBackgroundOn()
                self.renderer.SetBackground2(getColor(bg2_))

        if axes_ is not None:
            self.axes = axes_

        if self.offscreen:
            interactive = False
            self.interactive = False

        self.resetcam = resetcam

        def scan(wannabeacts):
            scannedacts = []
            if not utils.isSequence(wannabeacts):
                wannabeacts = [wannabeacts]

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

                elif isinstance(a, vtk.vtkAssembly):
                    scannedacts.append(a)
                    import vedo.pyplot as pyplot
                    if isinstance(a, pyplot.Plot):
                        a.modified = False
                        self.sharecam = False
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

                elif isinstance(a, vedo.Volume):
                    scannedacts.append(a)

                elif isinstance(a, vedo.TetMesh):
                    # check ugrid is all made of tets
                    ugrid = a.inputdata()
                    uarr = ugrid.GetCellTypesArray()
                    celltypes = np.unique(vtk_to_numpy(uarr))
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

                elif isinstance(a, vtk.vtkVolume): # order matters!
                    scannedacts.append(vedo.Volume(a.GetMapper().GetInput()))

                elif isinstance(a, vtk.vtkImageActor):
                    scannedacts.append(a)

                elif isinstance(a, vtk.vtkImageData):
                    scannedacts.append(vedo.Volume(a))

                elif isinstance(a, vtk.vtkBillboardTextActor3D):
                    scannedacts.append(a)

                elif isinstance(a, str):  # assume a filepath or 2D comment was given
                    import os.path
                    if "." in a and ". " not in a and os.path.isfile(a):
                        out = vedo.io.load(a)
                    else:
                        from vedo.shapes import Text2D
                        out = Text2D(a, pos=3)
                    scannedacts.append(out)

                elif isinstance(a, vtk.vtkMultiBlockDataSet):
                    for i in range(a.GetNumberOfBlocks()):
                        b =  a.GetBlock(i)
                        if isinstance(b, vtk.vtkPolyData):
                            scannedacts.append(vedo.Mesh(b))
                        elif isinstance(b, vtk.vtkImageData):
                            scannedacts.append(vedo.Volume(b))

                elif "dolfin" in str(type(a)):  # assume a dolfin.Mesh object
                    from vedo.dolfin import MeshActor
                    scannedacts.append(MeshActor(a))

                elif "trimesh" in str(type(a)):
                    from vedo.utils import trimesh2vedo
                    scannedacts.append(trimesh2vedo(a))

                else:
                    try:
                        scannedacts.append(vedo.Mesh(a))
                    except:
                        printc("Cannot understand input in show():", type(a), c='r')
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
        if settings.notebookBackend and settings.notebookBackend != "panel" and settings.notebookBackend != "2d":
            return backends.getNotebookBackend(actors2show, zoom, viewup)
        #########################################################################

        if not hasattr(self, 'window'):
            return self

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
                return self

        if at is None:
            at = 0

        if at < len(self.renderers):
            self.renderer = self.renderers[at]
        else:
            if settings.notebookBackend:
                printc("Error in show(): multiple renderings not supported in notebooks.", c='r')
            else:
                printc("Error in show(): wrong renderer index", at, c='r')
            return self

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

                if hasattr(ia, '_set2actcam') and ia._set2actcam:
                    ia.SetCamera(self.camera)

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
                            breppr.SetColor(getColor(settings.flagColor))
                            breppr.SetBackgroundColor(getColor(settings.flagBackgroundColor))
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
                    elif isinstance(ia.scalarbar, vedo.Assembly):
                        for a in ia.scalarbar.unpack():
                            self.renderer.RemoveActor(a)
                if hasattr(ia, 'renderedAt'):
                    ia.renderedAt.discard(at)


        if self.axes is not None:
            if viewup != "2d" or self.axes in [1, 8] or isinstance(self.axes, dict):
                addons.addGlobalAxes(self.axes)

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
            self.interactor.Initialize()
            self.initializedIren = True
            self.interactor.RemoveObservers("CharEvent")

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
            elif viewup == "2d":
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
                if not self.interactive: self.interactor.Start()
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
        pos = options.pop("pos", 0)
        size = options.pop("size", 0.1)
        c = options.pop("c", 'r')
        draggable = options.pop("draggable", True)

        if not self.renderer:
            printc("Use showInset() after first rendering the scene.", c='y')
            save_int = self.interactive
            self.show(interactive=0)
            self.interactive = save_int
        widget = vtk.vtkOrientationMarkerWidget()
        r,g,b = getColor(c)
        widget.SetOutlineColor(r,g,b)
        if len(actors)==1:
            widget.SetOrientationMarker(actors[0])
        else:
            widget.SetOrientationMarker(vedo.Assembly(utils.flatten(actors)))

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
        self.interactor.Render()
        return widget


    def clear(self, actors=None):
        """Delete specified list of actors, by default delete all."""
        if actors is None:
            self.renderer.RemoveAllViewProps()
            self.actors = []
            settings.collectable_actors = []
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
        self.clear()
        self.closeWindow()
        self.actors = []
        settings.collectable_actors = []
        settings.plotter_instance = None

    def screenshot(self, filename):
        vedo.io.screenshot(filename)
        return self


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


    #######################################################################
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
        self.picked3d = picker.GetPickPosition()

        if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
            return
        self.clickedActor = clickedActor

        if self.mouseMiddleClickFunction:
            self.mouseMiddleClickFunction(self.clickedActor)


    #######################################################################
    def _keypress(self, iren, event):
        # qt creates and passes a vtkGenericRenderWindowInteractor

        key = iren.GetKeySym()
        #print('Pressed key:', key)

        if key in ["q", "space", "Return"]:
            iren.ExitCallback()
            return

        elif key == "Escape":
            sys.stdout.flush()
            settings.plotter_instance.closeWindow()

        elif key in ["F1", "Pause"]:
            sys.stdout.flush()
            printc('\n[F1] pressed. Execution aborted. Exiting python now.', c='r')
            settings.plotter_instance.close()
            sys.exit(0)


        #############################################################
        ### now intercept custom observer ###########################
        #############################################################
        if self.keyPressFunction:
            if key not in ["Shift_L", "Control_L", "Super_L", "Alt_L"]:
                if key not in ["Shift_R", "Control_R", "Super_R", "Alt_R"]:
                    self.verbose = False
                    self.keyPressFunction(key)
                    return

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
            from vedo.docs import tips
            tips()
            return

        elif key == "a":
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

        elif key == "j":
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

        elif key == "S":
            vedo.io.screenshot("screenshot.png")
            printc("\camera Saved rendering window as screenshot.png", c="blue")
            return

        elif key == "C":
            cam = self.renderer.GetActiveCamera()
            printc('\n###################################################', c='y')
            printc('### Template python code to position this camera: ###', c='y')
            printc('cam = dict(pos='          +utils.precision(cam.GetPosition(),3)+',', c='y')
            printc('           focalPoint='   +utils.precision(cam.GetFocalPoint(),3)+',', c='y')
            printc('           viewup='       +utils.precision(cam.GetViewUp(),3)+',', c='y')
            printc('           distance='     +utils.precision(cam.GetDistance(),3)+',', c='y')
            printc('           clippingRange='+utils.precision(cam.GetClippingRange(),3)+')', c='y')
            printc('show(mymeshes, camera=cam)', c='y')
            printc('\n### OR equivalently: ##############################', c='y')
            printc('plt = vedo.Plotter()\n...', c='y')
            printc('plt.camera.SetPosition(',   [round(e, 3) for e in cam.GetPosition()],  ')', c='y')
            printc('plt.camera.SetFocalPoint(', [round(e, 3) for e in cam.GetFocalPoint()], ')', c='y')
            printc('plt.camera.SetViewUp(',     [round(e, 3) for e in cam.GetViewUp()], ')', c='y')
            printc('plt.camera.SetDistance(',   round(cam.GetDistance(), 3), ')', c='y')
            printc('plt.camera.SetClippingRange(',
                                    [round(e, 3) for e in cam.GetClippingRange()], ')', c='y')
            printc('###################################################', c='y')
            return

        elif key == "s":
            if self.clickedActor and self.clickedActor in self.getMeshes():
                self.clickedActor.GetProperty().SetRepresentationToSurface()
            else:
                for a in self.getMeshes():
                    if a and a.GetPickable():
                        a.GetProperty().SetRepresentationToSurface()

        elif key == "1":
            self.icol += 1
            if self.clickedActor and hasattr(self.clickedActor, "GetProperty"):
                self.clickedActor.GetMapper().ScalarVisibilityOff()
                self.clickedActor.GetProperty().SetColor(vedo.colors.colors1[(self.icol) % 10])
            else:
                for i, ia in enumerate(self.getMeshes()):
                    if not ia.GetPickable():
                        continue
                    ia.GetProperty().SetColor(vedo.colors.colors1[(i + self.icol) % 10])
                    ia.GetMapper().ScalarVisibilityOff()
            addons.addLegend()

        elif key == "2":
            self.icol += 1
            if self.clickedActor and hasattr(self.clickedActor, "GetProperty"):
                self.clickedActor.GetMapper().ScalarVisibilityOff()
                self.clickedActor.GetProperty().SetColor(vedo.colors.colors2[(self.icol) % 10])
            else:
                for i, ia in enumerate(self.getMeshes()):
                    if not ia.GetPickable():
                        continue
                    ia.GetProperty().SetColor(vedo.colors.colors2[(i + self.icol) % 10])
                    ia.GetMapper().ScalarVisibilityOff()
            addons.addLegend()

        elif key == "3":
            c = getColor("gold")
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
                if isinstance(ia, vedo.pointcloud.Points):
                    arnames = ia.getArrayNames()['PointData']
                    if len(arnames):
                        arnam =  arnames[ia._scals_idx]
                        if arnam and "normals" not in arnam.lower(): # exclude normals
                            ia.getPointArray( ia._scals_idx )
                            printc("..active scalars set to:", arnam, c='g', bold=0)
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
            bg2name = vedo.colors.getColorName(self.renderer.GetBackground2())
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
                self.renderer.SetBackground2(getColor(bg2name_next))

        elif key in ["plus", "equal", "KP_Add", "minus", "KP_Subtract"]:  # cycle axes style
            clickedr = self.renderers.index(self.renderer)
            if self.axes_instances[clickedr]:
                if hasattr(self.axes_instances[clickedr], "EnabledOff"):  # widget
                    self.axes_instances[clickedr].EnabledOff()
                else:
                    self.renderer.RemoveActor(self.axes_instances[clickedr])
                self.axes_instances[clickedr] = None
            if not self.axes: self.axes=0
            if key in ["minus", "KP_Subtract"]:
                addons.addGlobalAxes(axtype=(self.axes-1)%14, c=None)
            else:
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
                        printc('-> lighting set to:', shds[lnr], c='g', bold=0)
                    except AttributeError:
                        pass

        elif key == "K": # shading
            if self.clickedActor in self.getMeshes():
                acts = [self.clickedActor]
            else:
                acts = self.getMeshes()
            for ia in acts:
                if ia.GetPickable() and isinstance(ia, vedo.Mesh):
                    ia.computeNormals()
                    intrp = (ia.GetProperty().GetInterpolation()+1)%3
                    ia.GetProperty().SetInterpolation(intrp)
                    printc('->  shading set to:',
                                  ia.GetProperty().GetInterpolationAsString(),
                                  c='g', bold=0)

        elif key == "n":  # show normals to an actor
            from vedo.shapes import NormalLines
            if self.clickedActor in self.getMeshes():
                if self.clickedActor.GetPickable():
                    self.renderer.AddActor(NormalLines(self.clickedActor))
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
                    printc("\save Saved file:", fname, c="m")
                    self.cutterWidget.Off()
                    self.cutterWidget = None
            else:
                for a in self.actors:
                    if isinstance(a, vtk.vtkVolume):
                        addons.addCutterTool(a)
                        return

                printc("Click object and press X to open the cutter box widget.", c=4)

        elif key == "E":
            printc("\camera Exporting 3D window to file", c="blue", end="")
            vedo.io.exportWindow('scene.npz')
            printc(". Try:\n> vedo scene.npz", c="blue")
            settings.plotter_instance.interactor.Start()

        elif key == "i":  # print info
            if self.clickedActor:
                utils.printInfo(self.clickedActor)
            else:
                utils.printInfo(self)

        if iren:
            iren.Render()

