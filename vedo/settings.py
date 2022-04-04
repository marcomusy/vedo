"""
General settings to modify the global behavior


Usage example
-------------
.. code-block:: python

    from vedo import *
    settings.useParallelProjection = True
    Cube().color('g').show()


Parameters
----------

    # Set a default for the font to be used for axes, comments etc.
    defaultFont = 'Normografo' # check font options in shapes.Text

    # Palette number when using an integer to choose a color
    palette = 0

    # Scale magnification of the screenshot (must be an integer)
    screeshotScale = 1
    screenshotTransparentBackground = False
    screeshotLargeImage = False # Sometimes setting this to True gives better results

    # Allow to continously interact with scene during interactive() execution
    allowInteraction = False

    # Set up default mouse and keyboard functionalities
    enableDefaultMouseCallbacks = True
    enableDefaultKeyboardCallbacks = True

    # If False, when multiple renderers are present do not render each one for separate
    #  but do it just once at the end (when interactive() is called)
    immediateRendering = True

    # Show a gray frame margin in multirendering windows
    rendererFrameColor = None
    rendererFrameAlpha = 0.5
    rendererFrameWidth = 0.5
    rendererFramePadding = 0.0001

    # In multirendering mode set the position of the horizontal of vertical splitting [0,1]
    windowSplittingPosition = None

    # Enable / disable color printing by printc()
    enablePrintColor = True

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
    multiSamples    = 8     # force to not pick a framebuffer with a multisample buffer
    maxNumberOfPeels= 4     # maximum number of rendering passes
    occlusionRatio  = 0.0   # occlusion ratio, 0 = exact image.

    # Turn on/off nvidia FXAA post-process anti-aliasing, if supported.
    useFXAA = False         # either True or False

    # By default, the depth buffer is reset for each renderer. If True, use the existing depth buffer
    preserveDepthBuffer = False

    # Turn on/off Screen Space Ambient Occlusion (SSAO), darken some pixels to improve depth perception
    useSSAO         = False
    SSAORadius      = 0.5   # the SSAO hemisphere radius
    SSAOBias        = 0.01  # the bias when comparing samples
    SSAOKernelSize  = 32    # the number of samples
    SSAOBlur        = False # blurring of the ambient occlusion (helps for low samples nr)

    # Use a polygon/edges offset to possibly resolve conflicts in rendering
    usePolygonOffset    = False
    polygonOffsetFactor = 0.1
    polygonOffsetUnits  = 0.1

    # Interpolate scalars to render them smoothly
    interpolateScalarsBeforeMapping = True

    # Set parallel projection On or Off (place camera to infinity, no perspective effects)
    useParallelProjection = False

    # Set orientation type when reading TIFF files (volumes):
    # TOPLEFT  1 (row 0 top, col 0 lhs)    TOPRIGHT 2 (row 0 top, col 0 rhs)
    # BOTRIGHT 3 (row 0 bottom, col 0 rhs) BOTLEFT  4 (row 0 bottom, col 0 lhs)
    # LEFTTOP  5 (row 0 lhs, col 0 top)    RIGHTTOP 6 (row 0 rhs, col 0 top)
    # RIGHTBOT 7 (row 0 rhs, col 0 bottom) LEFTBOT  8 (row 0 lhs, col 0 bottom)
    tiffOrientationType = 1

    # AnnotatedCube axis type nr. 5 options:
    annotatedCubeColor      = (0.75, 0.75, 0.75)
    annotatedCubeTextColor  = None # use default, otherwise specify a single color
    annotatedCubeTextScale  = 0.2
    annotatedCubeTexts      = ["right","left ", "front","back ", " top ", "bttom"]

    # k3d settings for jupyter notebooks
    k3dMenuVisibility = True
    k3dPlotHeight = 512
    k3dAntialias  = True
    k3dLighting   = 1.2
    k3dCameraAutoFit = True
    k3dGridAutoFit= True
    k3dAxesHelper = True    # size of the small triad of axes on the bottom right
    k3dPointShader= "mesh"  # others are '3d', '3dSpecular', 'dot', 'flat'
    k3dLineShader = "thick" # others are 'flat', 'mesh'
"""

defaultFont = 'Normografo'

# Palette number when using an integer to choose a color
palette = 0

# last_figure = None

# Scale magnification of the screenshot (must be an integer)
screeshotScale = 1
screenshotTransparentBackground = False
screeshotLargeImage = False

# Allow to continously interact with scene during interactor.Start() execution
allowInteraction = False

# BUG in vtk9.0 (if true close works but sometimes vtk crashes, if false doesnt crash but cannot close)
# see plotter.py line 555
hackCallScreenSize = True

# Set up default mouse and keyboard functionalities
enableDefaultMouseCallbacks = True
enableDefaultKeyboardCallbacks = True

# When multiple renderers are present do not render each one for separate.
# but do it just once at the end (when interactive() is called)
immediateRendering = True

# Show a gray frame margin in multirendering windows
rendererFrameColor = None
rendererFrameAlpha = 0.5
rendererFrameWidth = 0.5
rendererFramePadding = 0.0001

# Wrap lines in tubes
# renderPointsAsSpheres has become mesh.renderPointsAsSpheres(True)
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
#print("vtk_version sys_platform", vtk_version, sys_platform)
useDepthPeeling = False
multiSamples = 8
#if vtk_version[0] >= 9: # moved to __init__
#    if "Windows" in sys_platform:
#        useDepthPeeling = True
# only relevant if depthpeeling is on
alphaBitPlanes   = 1
maxNumberOfPeels = 4
occlusionRatio   = 0.1

# Turn on/off nvidia FXAA anti-aliasing, if supported.
useFXAA = False  # either True or False

# By default, the depth buffer is reset for each renderer. If true, use the existing depth buffer
preserveDepthBuffer = False

#Enable or disable Screen Space Ambient Occlusion: SSAO darkens some pixels to improve depth perception.
useSSAO        = False
SSAORadius     = 0.5     # define the SSAO hemisphere radius
SSAOBias       = 0.01    # define the bias when comparing samples
SSAOKernelSize = 32      # define the number of samples
SSAOBlur       = False   # define blurring of the ambient occlusion (helps for low samples)

# Use a polygon/edges offset to possibly resolve conflicts in rendering
usePolygonOffset = True
polygonOffsetFactor = 0.1
polygonOffsetUnits  = 0.1

# Interpolate scalars to render them smoothly
interpolateScalarsBeforeMapping = True

# Set parallel projection On or Off (place camera to infinity, no perspective effects)
useParallelProjection = False

# In multirendering mode set the position of the horizontal of vertical splitting [0,1]
windowSplittingPosition = None

# Set orientation type when reading TIFF files (volumes):
# TOPLEFT  1 (row 0 top, col 0 lhs)    TOPRIGHT 2 (row 0 top, col 0 rhs)
# BOTRIGHT 3 (row 0 bottom, col 0 rhs) BOTLEFT  4 (row 0 bottom, col 0 lhs)
# LEFTTOP  5 (row 0 lhs, col 0 top)    RIGHTTOP 6 (row 0 rhs, col 0 top)
# RIGHTBOT 7 (row 0 rhs, col 0 bottom) LEFTBOT  8 (row 0 lhs, col 0 bottom)
tiffOrientationType = 1

# AnnotatedCube axis (type 5) customization:
annotatedCubeColor      = (0.75, 0.75, 0.75)
annotatedCubeTextColor  = None # use default, otherwise specify a single color
annotatedCubeTextScale  = 0.2
annotatedCubeTexts      = ["right","left ", "front","back ", " top ", "bttom"]

# enable / disable color printing
enablePrintColor = True

####################################################################################
# k3d settings for jupyter notebooks
k3dMenuVisibility = True
k3dPlotHeight = 512
k3dAntialias  = True
k3dLighting   = 1.2
k3dCameraAutoFit = True
k3dGridAutoFit= True
k3dAxesHelper = True    # size of the small triad of axes on the bottom right
k3dPointShader= "mesh"  # others are '3d', '3dSpecular', 'dot', 'flat'
k3dLineShader = "thick" # others are 'flat', 'mesh'

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


####################################################################################
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
                        islocal = False,
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
        Meson = dict(
                        mono = False,
                        fscale = 0.8,
                        hspacing = 0.9,
                        lspacing = 0.225,
                        dotsep = "~^.~ ",
                        islocal = False,
                        ),
        Komika = dict(
                        mono = False,
                        fscale = 0.7,
                        hspacing = 0.75,
                        lspacing = 0.225,
                        dotsep = "~^.~ ",
                        islocal = False,
                        ),
        Vogue = dict(
                        mono = False,
                        fscale = 0.7,
                        hspacing = 0.75,
                        lspacing = 0.225,
                        dotsep = "~^.~ ",
                        islocal = False,
                        ),
        Brachium = dict(
                        mono = True,
                        fscale = 0.8,
                        hspacing = 1,
                        lspacing = 0.1,
                        dotsep = "·",
                        islocal = False,
                        ),
        Dalim = dict(
                        mono = False,
                        fscale = 0.75,
                        lspacing = 0.2,
                        hspacing = 1,
                        dotsep = '~·',
                        islocal = False,
                        ),
        Miro = dict(
                        mono = False,
                        fscale = 0.75,
                        lspacing = 0.2,
                        hspacing = 1,
                        dotsep = '~·',
                        islocal = False,
                        ),
        Ubuntu = dict(
                        mono = False,
                        fscale = 0.75,
                        lspacing = 0.2,
                        hspacing = 1,
                        dotsep = '~·',
                        islocal = False,
                        ),
)


