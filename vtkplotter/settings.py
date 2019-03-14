"""
Global settings.
"""
from __future__ import division, print_function

__all__ = [
    'computeNormals',
    'interactorStyle',
    'allowInteraction',
    'usingQt',
    'enableDolfin',
    'renderPointsAsSpheres',
]

# recompute vertex and cell normals
computeNormals = None

# default style is TrackBallCamera
interactorStyle = None

# allow to interact with scene during interactor.Start() execution
allowInteraction = True

# Qt embedding
usingQt = False

# if disabled, the whole package imports a bit faster
enableDolfin = False

# on some vtk versions/platforms points are redered as ugly squares
renderPointsAsSpheres = True

# sync different Plotter instances
syncPlotters = True

#####################
def _init():
    global plotter_instance, plotter_instances, collectable_actors
    plotter_instance = None
    plotter_instances = []
    collectable_actors = []
    
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)
