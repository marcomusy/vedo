"""
General settings.

.. code-block:: python

    # Axes title defaults
    xtitle = 'x'
    ytitle = 'y'
    ztitle = 'z'

    # Set a default for the font to be used for axes, comments etc.
    defaultFont = 'Normografo' # check font options in shapes.Text

    # Scale magnification of the screenshot (must be an integer)
    screeshotScale = 1
    screenshotTransparentBackground = False
    screeshotLargeImage = False # Sometimes setting this to True gives better results

    # Recompute vertex and cell normals
    computeNormals = None

    # Automatically reset the range of the active scalars
    autoResetScalarRange = True

    # Default style is TrackBallCamera
    interactorStyle = None
    # possible values are (see https://vtk.org/doc/nightly/html/classvtkInteractorStyle.html):
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

    # Allow to interact with scene during interactor.Start() execution
    allowInteraction = True

    # If False, when multiple renderers are present do not render each one for separate
    #  but do it just once at the end (when interactive() is called)
    immediateRendering = True

    # Show a gray frame margin in multirendering windows
    showRendererFrame = True
    rendererFrameColor = None

    # In multirendering mode set the position of the horizontal of vertical splitting [0,1]
    windowSplittingPosition = None

    # Use tex, matplotlib latex compiler
    usetex = False

    # Enable / disable color printing by printc()
    enablePrintColor = True

    # Qt embedding
    usingQt = False

    # OpenVR rendering (untested)
    useOpenVR = False

    # Wrap lines in tubes
    renderLinesAsTubes = False

    # Smoothing options
    pointSmoothing = False
    lineSmoothing = False
    polygonSmoothing = False

    # Remove hidden lines when in wireframe mode
    hiddenLineRemoval = False

    # For Structured and RectilinearGrid: show internal edges not only outline
    visibleGridEdges = False

    # Turn on/off the automatic repositioning of lights as the camera moves.
    lightFollowsCamera = False
    twoSidedLighting = True

    # Turn on/off rendering of translucent material with depth peeling technique.
    useDepthPeeling = False
    alphaBitPlanes  = True  # options only active if useDepthPeeling=True
    multiSamples    = 0     # force to not pick a framebuffer with a multisample buffer
    maxNumberOfPeels= 8     # maximum number of rendering passes
    occlusionRatio  = 0.0   # occlusion ratio, 0 = exact image.

    # Turn on/off nvidia FXAA anti-aliasing, if supported.
    useFXAA = None          # either True or False. None sets the system default

    # Use a polygon/edges offset to possibly resolve conflicts in rendering
    usePolygonOffset    = False
    polygonOffsetFactor = 0.1
    polygonOffsetUnits  = 0.1

    # Interpolate scalars to render them smoothly
    interpolateScalarsBeforeMapping = True

    # Set parallel projection On or Off (place camera to infinity, no perspective effects)
    useParallelProjection = False

    # AnnotatedCube axis type nr. 5 options:
    annotatedCubeColor      = (0.75, 0.75, 0.75)
    annotatedCubeTextColor  = None # use default, otherwise specify a single color
    annotatedCubeTextScale  = 0.2
    annotatedCubeXPlusText  = "right"
    annotatedCubeXMinusText = "left "
    annotatedCubeYPlusText  = "front"
    annotatedCubeYMinusText = "back "
    annotatedCubeZPlusText  = " top "
    annotatedCubeZMinusText = "bttom"

Usage example:

.. code-block:: python

    from vedo import *

    settings.useParallelProjection = True

    Cube().color('green').show()

"""
import os, vtk

__all__ = ['datadir', 'dataurl', 'embedWindow']


####################################################################################
# Axes titles
xtitle = 'x'
ytitle = 'y'
ztitle = 'z'
defaultAxesType = None

# Set a default for the font to be used for axes, comments etc.
defaultFont = 'Normografo'

# Scale magnification of the screenshot (must be an integer)
screeshotScale = 1
screenshotTransparentBackground = False
screeshotLargeImage = False

# Recompute vertex and cell normals
computeNormals = None

# Automatic resetting the range of the active scalars
autoResetScalarRange = True

# Default style is TrackBallCamera
interactorStyle = None

# Allow to interact with scene during interactor.Start() execution
allowInteraction = True

# When multiple renderers are present do not render each one for separate.
# but do it just once at the end (when interactive() is called)
immediateRendering = True

# Show a gray frame margin in multirendering windows
showRendererFrame = True
rendererFrameColor = None
rendererFrameAlpha = 0.5
rendererFrameWidth = 0.5

# Use tex, matplotlib latex compiler
usetex = False

# Qt embedding
usingQt = False

# OpenVR rendering
useOpenVR = False

# Wrap lines in tubes
renderLinesAsTubes = False

# Remove hidden lines when in wireframe mode
hiddenLineRemoval = False

# Smoothing options
pointSmoothing = False
lineSmoothing = False
polygonSmoothing = False

# For Structured and RectilinearGrid: show internal edges not only outline
visibleGridEdges = False

# Turn on/off the automatic repositioning of lights as the camera moves.
lightFollowsCamera = False
twoSidedLighting = True

# Turn on/off rendering of translucent material with depth peeling technique.
#https://lorensen.github.io/VTKExamples/site/Cxx/Visualization/CorrectlyRenderTranslucentGeometry
useDepthPeeling = False
if '9' in vtk.vtkVersion().GetVTKVersion():
    useDepthPeeling = True
alphaBitPlanes  = True  # only relevant if useDepthPeeling
multiSamples    = 0
maxNumberOfPeels= 8
occlusionRatio  = 0.0

# Turn on/off nvidia FXAA anti-aliasing, if supported.
useFXAA = None           # either True or False. None sets the system default

# Use a polygon/edges offset to possibly resolve conflicts in rendering
usePolygonOffset = False
polygonOffsetFactor = 0.1
polygonOffsetUnits  = 0.1

# Interpolate scalars to render them smoothly
interpolateScalarsBeforeMapping = True

# Set parallel projection On or Off (place camera to infinity, no perspective effects)
useParallelProjection = False

# In multirendering mode set the position of the horizontal of vertical splitting [0,1]
windowSplittingPosition = None

# AnnotatedCube axis (type 5) customization:
annotatedCubeColor      = (0.75, 0.75, 0.75)
annotatedCubeTextColor  = None # use default, otherwise specify a single color
annotatedCubeTextScale  = 0.2
annotatedCubeXPlusText  = "right"
annotatedCubeXMinusText = "left "
annotatedCubeYPlusText  = "front"
annotatedCubeYMinusText = "back "
annotatedCubeZPlusText  = " top "
annotatedCubeZMinusText = "bttom"

# enable / disable color printing
enablePrintColor = True

####################################################################################
# notebook support with K3D
notebookBackend = None
notebook_plotter = None

####################################################################################
flagDelay = 150 # values will be superseded
flagFont = "Courier"
flagFontSize = 18
flagJustification = 0
flagAngle = 0
flagBold = False
flagItalic = False
flagShadow = False
flagColor = 'k'
flagBackgroundColor = 'w'


#############################
installdir = os.path.dirname(__file__)

textures_path = os.path.join(installdir, "textures/")
textures = []

fonts_path = os.path.join(installdir, "fonts/")
fonts = []

#dataurl = "/home/musy/Dropbox/Public/vktwork/vedo_data/"; print('\ndataurl=',dataurl)
dataurl = "https://vedo.embl.es/examples/data/"
datadir = dataurl

plotter_instances = []
plotter_instance = None

####################################################################################
# mono       # means that all letters occupy the same space slot horizontally
# hspacing   # an horizontal stretching factor (affects both letters and words)
# lspacing   # horizontal spacing inbetween letters (not words)
# islocal    # is locally stored in /fonts, otherwise it's on vedo.embl.es/fonts

font_parameters = dict(

        Normografo = dict(
                        mono = False,
                        fscale = 0.75,
                        hspacing = 1,
                        lspacing = 0.2,
                        dotsep = "~·",
                        islocal = True,
                        ),
        Bongas = dict(
                        mono = False,
                        fscale = 0.875,
                        hspacing = 0.52,
                        lspacing = 0.25,
                        dotsep = "·",
                        islocal = True,
                        ),
        Calco = dict(
                        mono = True,
                        fscale = 0.8,
                        hspacing = 1,
                        lspacing = 0.1,
                        dotsep = "·",
                        islocal = True,
                        ),
        Comae = dict(
                        mono = False,
                        fscale = 0.75,
                        lspacing = 0.2,
                        hspacing = 1,
                        dotsep = '~·',
                        islocal = True,
                        ),
        Glasgo = dict(
                        mono = True,
                        fscale = 0.75,
                        lspacing = 0.1,
                        hspacing = 1,
                        dotsep = "·",
                        islocal = True,
                        ),
        Kanopus = dict(
                        mono = False,
                        fscale = 0.75,
                        lspacing = 0.15,
                        hspacing = 0.75,
                        dotsep = '~·',
                        islocal = True,
                        ),
        LionelOfParis = dict(
                        mono = False,
                        fscale = 0.875,
                        hspacing = 0.7,
                        lspacing = 0.3,
                        dotsep = "·",
                        islocal = True,
                        ),
        LogoType = dict(
                        mono = False,
                        fscale = 0.75,
                        hspacing = 1,
                        lspacing = 0.2,
                        dotsep = '·~~',
                        islocal = False,
                        ),
        Quikhand = dict(
                        mono = False,
                        fscale = 0.8,
                        hspacing = 0.6,
                        lspacing = 0.15,
                        dotsep = "~~·~",
                        islocal = True,
                        ),
        SmartCouric = dict(
                        mono = True,
                        fscale = 0.8,
                        hspacing = 1.05,
                        lspacing = 0.1,
                        dotsep = "·",
                        islocal = True,
                        ),
        Spears = dict(
                        mono = False,
                        fscale = 0.8,
                        hspacing = 0.5,
                        lspacing = 0.2,
                        dotsep = "·",
                        islocal = False,
                        ),
        Theemim = dict(
                        mono = False,
                        fscale = 0.825,
                        hspacing = 0.52,
                        lspacing = 0.3,
                        dotsep = '~·',
                        islocal = True,
                        ),
        VictorMono = dict(
                        mono = True,
                        fscale = 0.725,
                        hspacing = 1,
                        lspacing = 0.1,
                        dotsep = "·",
                        islocal = True,
                        ),
        Justino1 = dict(
                        mono = True,
                        fscale = 0.725,
                        hspacing = 1,
                        lspacing = 0.1,
                        dotsep = "·",
                        islocal = False,
                        ),
        Justino2 = dict(
                        mono = True,
                        fscale = 0.725,
                        hspacing = 1,
                        lspacing = 0.1,
                        dotsep = "·",
                        islocal = False,
                        ),
        Justino3 = dict(
                        mono = True,
                        fscale = 0.725,
                        hspacing = 1,
                        lspacing = 0.1,
                        dotsep = "·",
                        islocal = False,
                        ),
        Justino4 = dict(
                        mono = True,
                        fscale = 0.725,
                        hspacing = 1,
                        lspacing = 0.1,
                        dotsep = "·",
                        islocal = False,
                        ),
        Capsmall = dict(
                        mono = False,
                        fscale = 0.8,
                        hspacing = 0.75,
                        lspacing = 0.15,
                        dotsep = "·",
                        islocal = False,
                        ),
        Cartoons123 = dict(
                        mono = False,
                        fscale = 0.8,
                        hspacing = 0.75,
                        lspacing = 0.15,
                        dotsep = "·",
                        islocal = False,
                        ),
        Vega = dict(
                        mono = False,
                        fscale = 0.8,
                        hspacing = 0.75,
                        lspacing = 0.15,
                        dotsep = "·",
                        islocal = False,
                        ),
        PlanetBenson = dict(
                        mono = False,
                        fscale = 0.8,
                        hspacing = 0.8,
                        lspacing = 0.11,
                        dotsep = "·",
                        islocal = False,
                        ),
        VTK = dict(
                        mono = False,
                        fscale = 0.8,
                        hspacing = 0.6,
                        lspacing = 0.4,
                        dotsep = "~^.~ ",
                        islocal = True,
                        ),
        Meson= dict(
                        mono = False,
                        fscale = 0.8,
                        hspacing = 0.9,
                        lspacing = 0.225,
                        dotsep = "~^.~ ",
                        islocal = False,
                        ),
)


####################################################################################
def embedWindow(backend='k3d', verbose=True):
    """Use this function to control whether the rendering window is inside
    the jupyter notebook or as an independent external window"""
    global notebook_plotter, notebookBackend

    if not backend:
        notebookBackend = None
        notebook_plotter = None
        return
    else:

        if any(['SPYDER' in name for name in os.environ]):
            notebookBackend = None
            notebook_plotter = None
            return

        try:
            get_ipython()
        except:
            notebookBackend = None
            notebook_plotter = None
            return

    notebookBackend = backend


    if backend=='k3d':
        try:
            import k3d
        except:
            notebookBackend = None
            if verbose:
                print('embedWindow(verbose=True): could not load k3d module, try:')
                print('> pip install k3d      # and if necessary:')
                print('> conda install nodejs')

    elif 'itk' in backend: # itkwidgets
        try:
            import itkwidgets
        except:
            notebookBackend = None
            if verbose:
                print('embedWindow(verbose=True): could not load itkwidgets module, try:')
                print('> pip install itkwidgets    # and if necessary:')
                print('> conda install nodejs')

    elif backend.lower() == '2d':
        verbose=False

    elif backend=='panel':
        try:
            import panel
            panel.extension('vtk')
        except:
            if verbose:
                print('embedWindow(verbose=True): could not load panel try:')
                print('> pip install panel -U   # and/or')
                print('> conda install nodejs')

    elif backend=='ipyvtk':
        try:
            from ipyvtk_simple.viewer import ViewInteractiveWidget
        except:
            if verbose:
                print('embedWindow(verbose=True): could not load ipyvtk_simple try:')
                print('> pip install -U git+https://github.com/Kitware/ipyvtk-simple.git')

    else:
        print("Unknown backend", backend)
        raise RuntimeError()


#####################
def _init():
    global plotter_instance, plotter_instances
    global textures, fonts
    global notebookBackend, notebook_plotter

    plotter_instance = None
    plotter_instances = []

    for f in os.listdir(textures_path):
        tfn = f.split(".")[0]
        if 'earth' in tfn: continue
        textures.append(tfn)

    for f in os.listdir(fonts_path):
        if '.npz' in f: continue
        fonts.append(f.split(".")[0])
    fonts = list(sorted(fonts))

    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    embedWindow()
