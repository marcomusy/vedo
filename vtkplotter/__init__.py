"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png

A python module for scientific visualization,
analysis and animation of 3D objects and point clouds based on VTK.

.. note:: **Please check out the** `git repository <https://github.com/marcomusy/vtkplotter-examples>`_.

    A full list of examples can be found in directories:

    - `examples/basic <https://github.com/marcomusy/vtkplotter-examples/blob/vtkplotter_master/examples/basic>`_ 
    - `examples/advanced <https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/advanced>`_ 
    - `examples/volumetric <https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/volumetric>`_
    - `examples/simulations <https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/simulations>`_
    - `examples/plotting2d <https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/plotting2d>`_
    - `examples/others <https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/other>`_
    - `examples/others/dolfin <https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/other/dolfin>`_.
    - `examples/others/trimesh <https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/other/trimesh>`_
"""
from __future__ import print_function

__author__ = "Marco Musy"
__license__ = "MIT"
__maintainer__ = "M. Musy, G. Dalmasso"
__email__ = "marco.musy@embl.es"
__status__ = "dev"
__website__ = "https://github.com/marcomusy/vtkplotter-examples"

from vtkplotter.version import _version as __version__
from vtkplotter.animation import Animation
from vtkplotter.plotter import *
from vtkplotter.analysis import *
from vtkplotter.plot2d import *
from vtkplotter.shapes import *
from vtkplotter.vtkio import *

from vtkplotter.base import ActorBase
from vtkplotter.assembly import Assembly
from vtkplotter.mesh import Mesh, merge, Actor # Actor is obsolete
from vtkplotter.picture import Picture
from vtkplotter.volume import Volume

from vtkplotter.utils import *
from vtkplotter.colors import *
import vtkplotter.settings as settings
from vtkplotter.settings import datadir, embedWindow

# hack: need to uncomment this to generate documentation html
from vtkplotter.dolfin import _inputsort

from numpy import sin, cos, sqrt, exp, log, dot, cross, array, arange

# imports hierarchy
# plotter : utils, colors, actors, vtkio, shapes
# analysis: utils, colors, actors, vtkio, shapes
# shapes  : utils, colors, actors
# vtkio   : utils, colors, actors
# actors  : utils, colors
# utils   :        colors
# colors  : -


###########################################################################
settings._init()
###########################################################################

## deprecations ############################################################
#def isolines(*args, **kargs):
#    printc("Obsolete. Use mesh.isolines() instead of isolines(mesh).", c=1)
#    raise RuntimeError()


