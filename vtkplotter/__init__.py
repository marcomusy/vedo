"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png

**A python class to easily draw, analyse and animate 3D objects and point clouds with VTK.**

.. note:: **Please check out the** `git repository <https://github.com/marcomusy/vtkplotter>`_.

    A full list of examples can be found in directories:
        
    - `examples/basic <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic>`_ ,
    - `examples/advanced <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced>`_ ,
    - `examples/volumetric <https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric>`_,
    - `examples/others <https://github.com/marcomusy/vtkplotter/blob/master/examples/other>`_.
"""

__author__     = "Marco Musy"
__license__    = "MIT"
__maintainer__ = "M. Musy, G. Dalmasso"
__email__      = "marco.musy@embl.es"
__status__     = "dev"
__website__    = "https://github.com/marcomusy/vtkplotter"
__version__    = "8.8.3" ### defined also in setup.py and docs/source/conf.py

from vtkplotter.plotter import *
from vtkplotter.analysis import *
from vtkplotter.shapes import *
from vtkplotter.vtkio import *
from vtkplotter.actors import *
from vtkplotter.utils import *
from vtkplotter.colors import *
import vtkplotter.settings as settings

from numpy import sin, cos, sqrt, exp, log, dot, cross, array, arange


def _ignore_warnings():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
_ignore_warnings()

# imports hierarchy
# plotter : utils, colors, actors, vtkio, shapes
# analysis: utils, colors, actors, vtkio, shapes
# shapes  : utils, colors, actors
# vtkio   : utils, colors, actors
# actors  : utils, colors
# utils   :        colors
# colors  : -

# basic/colormaps.py
# 