"""
Global settings.

.. code-block:: python

    # Axes titles
    xtitle = 'x'
    ytitle = 'y'
    ztitle = 'z'

    # Scale magnification of the screenshot (must be an integer)
    screeshotScale = 1
    screenshotTransparentBackground = False

    # Recompute vertex and cell normals
    computeNormals = None

    # Default style is TrackBallCamera
    interactorStyle = None

    # Allow to interact with scene during interactor.Start() execution
    allowInteraction = True

    # Show a gray frame margin in multirendering windows
    showRendererFrame = True
    rendererFrameColor = None

    # Use tex, matplotlib latex compiler
    usetex = False

    # Qt embedding
    usingQt = False

    # OpenVR rendering
    useOpenVR = False

    # On some vtk versions/platforms points are redered as ugly squares
    renderPointsAsSpheres = True

    # Wrap lines in tubes
    renderLinesAsTubes = False

    # Remove hidden lines when in wireframe mode
    hiddenLineRemoval = False

    # For (Un)Structured and RectilinearGrid: show internal edges not only outline
    visibleGridEdges = False

    # Turn on/off the automatic repositioning of lights as the camera moves.
    lightFollowsCamera = False

    # Turn on/off nvidia FXAA anti-aliasing, if supported.
    useFXAA = False

    # Turn on/off rendering of translucent material with depth peeling technique.
    useDepthPeeling = False

    # Set parallel projection On or Off (place camera to infinity, no perspective effects)
    useParallelProjection = False

    # Path to Voro++ library, http://math.lbl.gov/voro++
    voro_path = '/usr/local/bin'


Usage example:

.. code-block:: python

    from vtkplotter import *

    settings.useParallelProjection = True

    Cube().color('green').show()

"""
__all__ = ['datadir', 'embedWindow']


####################################################################################
# Axes titles
xtitle = 'x'
ytitle = 'y'
ztitle = 'z'

# Scale magnification of the screenshot (must be an integer)
screeshotScale = 1
screenshotTransparentBackground = False

# Recompute vertex and cell normals
computeNormals = None

# Default style is TrackBallCamera
interactorStyle = None

# Allow to interact with scene during interactor.Start() execution
allowInteraction = True

# Show a gray frame margin in multirendering windows
showRendererFrame = True
rendererFrameColor = None

# Use tex, matplotlib latex compiler
usetex = False

# Qt embedding
usingQt = False

# OpenVR rendering
useOpenVR = False

# On some vtk versions/platforms points are redered as ugly squares
renderPointsAsSpheres = True

# Wrap lines in tubes
renderLinesAsTubes = False

# Remove hidden lines when in wireframe mode
hiddenLineRemoval = False

# For (Un)Structured and RectilinearGrid: show internal edges not only outline
visibleGridEdges = False

# Turn on/off the automatic repositioning of lights as the camera moves.
lightFollowsCamera = False

# Turn on/off nvidia FXAA anti-aliasing, if supported.
useFXAA = False

# Turn on/off rendering of translucent material with depth peeling technique.
useDepthPeeling = False

# Set parallel projection On or Off (place camera to infinity, no perspective effects)
useParallelProjection = False

# Path to Voro++ library, http://math.lbl.gov/voro++
voro_path = '/usr/local/bin'




####################################################################################
# notebook support with K3D
notebookBackend = None
notebook_plotter = None

####################################################################################
import os
_cdir = os.path.dirname(__file__)
if _cdir == "":
    _cdir = "."
textures_path = _cdir + "/textures/"
textures = []

datadir = _cdir + "/data/"
fonts_path = _cdir + "/fonts/"
fonts = []

def embedWindow(backend='k3d', verbose=True):
    """Use this function to control whether the rendering window is inside
    the jupyter notebook or as an independent external window"""
    global notebook_plotter, notebookBackend

    if not backend:
        notebookBackend = None
        notebook_plotter = None
        return
    else:
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

    elif backend=='panel':
        try:
            if verbose:
                print('INFO: embedWindow(verbose=True), first import of panel module, this takes time...')
            import panel
            panel.extension('vtk')
        except:
            if verbose:
                print('embedWindow(verbose=True): could not load panel try:')
                print('> pip install panel    # and/or')
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
        textures.append(f.split(".")[0])
    textures.remove("earth")
    textures = list(sorted(textures))

    for f in os.listdir(fonts_path):
        fonts.append(f.split(".")[0])
    fonts.remove("licenses")
    fonts = list(sorted(fonts))

    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    embedWindow()



