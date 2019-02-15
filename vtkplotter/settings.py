'''
Global settings.
'''

__all__ = [
    'computeNormals',
    'interactorStyle',
    'allowInteraction',
    'usingQt',
]

# recompute vertex and cell normals
computeNormals = None

# default style is TrackBallCamera
interactorStyle = None

# allow to interact with scene during interactor.Start() execution
allowInteraction = True

# Qt embedding
usingQt = False



#####################
def _init():
    global plotter_instance, plotter_instances
    plotter_instance = None
    plotter_instances = []

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

