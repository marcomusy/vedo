'''
Global settings.
'''

__all__ = [
      'computeNormals',
      'interactorStyle',
       'allowInteraction',
       'usingQt',
       ]



def init():
    global plotter_instance
    plotter_instance = None

# recompute vertex and cell normals
computeNormals = None 

# default style is TrackBallCamera
interactorStyle = None 

# allow to interact with scene during interactor.Start() execution
allowInteraction = True 

# Qt embedding
usingQt = False

