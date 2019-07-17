"""
Global settings.

.. role:: raw-html-m2r(raw)
   :format: html

.. note:: **Please check out the** `git repository <https://github.com/marcomusy/vtkplotter>`_.

    A full list of examples can be found in directories:

    - `examples/basic <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic>`_ ,
    - `examples/advanced <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced>`_ ,
    - `examples/volumetric <https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric>`_,
    - `examples/simulations <https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations>`_.
    - `examples/other <https://github.com/marcomusy/vtkplotter/blob/master/examples/other>`_
    - `examples/other/dolfin <https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin>`_.

:raw-html-m2r:`<br />`

.. image:: https://user-images.githubusercontent.com/32848391/51558920-ec436e00-1e80-11e9-9d96-aa9b7c72d58b.png

:raw-html-m2r:`<br />`
:raw-html-m2r:`<br />`

"""
__all__ = ['datadir', 'embedWindow']

####################################################################################
# recompute vertex and cell normals
computeNormals = None

# default style is TrackBallCamera
interactorStyle = None

# allow to interact with scene during interactor.Start() execution
allowInteraction = True

# usetex, matplotlib latex compiler
usetex = False

# Qt embedding
usingQt = False

# OpenVR
useOpenVR = False

# on some vtk versions/platforms points are redered as ugly squares
renderPointsAsSpheres = True

renderLinesAsTubes = False

# remove hidden lines when in wireframe mode
hiddenLineRemoval = False

#
visibleGridEdges = False

# notebook support with K3D
notebookBackend = None
notebook_plotter = None

# path to Voro++ library
# http://math.lbl.gov/voro++
voro_path = '/usr/local/bin'

# axes titles
xtitle = 'x'
ytitle = 'y'
ztitle = 'z'

# scale magnification of the screenshot
screeshotScale = 1
screenshotTransparentBackground = False


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
    global notebook_plotter, notebookBackend

    if not backend:
        notebookBackend = None
        notebook_plotter = None
        return

    notebookBackend = backend
    if backend=='k3d':

        try:
            get_ipython()
        except:
            notebookBackend = None
            return

        try:
            import k3d
            #if verbose:
            #    print('INFO: embedWindow(verbose=True), importing k3d module')
        except:
            notebookBackend = None
            if verbose:
                print('embedWindow(verbose=True): could not load k3d module, try:')
                print('> pip install k3d      # and/or')
                print('> conda install nodejs')

    elif backend=='panel':
        try:
            get_ipython()
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



