"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png
   :alt: tea1

A python class to easily draw, analyse and animate 3D objects and point clouds with VTK.

Check out the `GitHub repository <https://github.com/marcomusy/vtkplotter>`_.
"""

__author__     = "Marco Musy"
__license__    = "MIT"
__maintainer__ = "M. Musy, G. Dalmasso"
__email__      = "marco.musy@embl.es"
__status__     = "dev"
__website__    = "https://github.com/marcomusy/vtkplotter"
__version__    = "8.8.0" ### defined also in setup.py and docs/source/conf.py

from vtkplotter.plotter import *
from vtkplotter.analysis import *
from vtkplotter.shapes import *
from vtkplotter.vtkio import *
from vtkplotter.actors import *
from vtkplotter.utils import *
from vtkplotter.colors import *

from numpy import sin, cos, sqrt, exp, log, dot, cross, array, arange


def _ignore_warnings():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
_ignore_warnings()

# imports
# plotter : utils, colors, actors, vtkio, shapes
# analysis:        colors, actors, vtkio, shapes
# shapes  : utils, colors, actors
# vtkio   : utils, colors, actors
# actors  : utils, colors
# utils   :        colors
# colors  : -