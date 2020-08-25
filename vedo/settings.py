"""
General settings.

.. code-block:: python

    # Axes title defaults
    xtitle = 'x'
    ytitle = 'y'
    ztitle = 'z'
    defaultAxesType = 4

    # Set a default for the font to be used for axes, comments etc.
    defaultFont = 'Normografo'
    # Options are:
    #    Biysk, Bongas, Calco, Comae, Kanopus, Glasgo, Inversionz, LionelOfParis,
    #    LogoType, Normografo, Quikhand, SmartCouric, Theemim, VictorMono, VTK.
    fontIsMono = True  # mono means that all letters occupy the same space slot horizontally
    fontHSpacing = 1   # an horizontal stretching factor (affects both letters and words)
    fontLSpacing = 0.1 # horizontal spacing inbetween letters (not words)

    # Scale magnification of the screenshot (must be an integer)
    screeshotScale = 1
    screenshotTransparentBackground = False

    # Sometimes setting this to True gives better results
    screeshotLargeImage = False

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

    # Show a gray frame margin in multirendering windows
    showRendererFrame = True
    rendererFrameColor = None

    # In multirendering mode set the position of the horizontal of vertical splitting [0,1]
    windowSplittingPosition = None

    # Use tex, matplotlib latex compiler
    usetex = False

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
import os

__all__ = ['datadir', 'embedWindow']


####################################################################################
# Axes titles
xtitle = 'x'
ytitle = 'y'
ztitle = 'z'
defaultAxesType = None

# Set a default for the font to be used for axes, comments etc.
defaultFont = 'Normografo'
fontIsMono = True
fontHSpacing = 1
fontLSpacing = 0.1

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

# Turn on/off rendering of translucent material with depth peeling technique.
#https://lorensen.github.io/VTKExamples/site/Cxx/Visualization/CorrectlyRenderTranslucentGeometry
useDepthPeeling = False
alphaBitPlanes  = True  # only active if useDepthPeeling
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

# AnnotatedCube axis type 5 customization:
annotatedCubeColor      = (0.75, 0.75, 0.75)
annotatedCubeTextColor  = None # use default, otherwise specify a single color
annotatedCubeTextScale  = 0.2
annotatedCubeXPlusText  = "right"
annotatedCubeXMinusText = "left "
annotatedCubeYPlusText  = "front"
annotatedCubeYMinusText = "back "
annotatedCubeZPlusText  = " top "
annotatedCubeZMinusText = "bttom"

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
flagItalic = True
flagShadow = False
flagColor = 'k'
flagBackgroundColor = 'w'
legendSize = 0.2
legendBC = (0.96, 0.96, 0.9)
legendPos = 2
legendFont = ""

installdir = os.path.dirname(__file__)

textures_path = os.path.join(installdir, "textures/")
textures = []

fonts_path = os.path.join(installdir, "fonts/")
fonts = []

#datadir = "/home/musy/Dropbox/Public/vktwork/vedo_data/"
datadir = "https://vedo.embl.es/examples/data/"

plotter_instances = []

plotter_instance = None
collectable_actors = []

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
    else:
        print("Unknown backend", backend)
        raise RuntimeError()


#####################
def _init():
    global plotter_instance, plotter_instances, collectable_actors
    global textures, fonts
    global notebookBackend, notebook_plotter

    plotter_instance = None
    plotter_instances = []
    collectable_actors = []

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
