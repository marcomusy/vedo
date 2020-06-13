"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png

A python module for scientific visualization,
analysis and animation of 3D objects and point clouds based on VTK.

.. note:: **Please check out the** `git repository <https://github.com/marcomusy/vedo>`_.

    A full list of examples can be found in directories:

    - `examples/basic <https://github.com/marcomusy/vedo/blob/master/vedo/examples/basic>`_
    - `examples/advanced <https://github.com/marcomusy/vedo/blob/master/vedo/examples/advanced>`_
    - `examples/volumetric <https://github.com/marcomusy/vedo/blob/master/vedo/examples/volumetric>`_
    - `examples/tetmesh <https://github.com/marcomusy/vedo/blob/master/vedo/examples/tetmesh>`_
    - `examples/simulations <https://github.com/marcomusy/vedo/blob/master/vedo/examples/simulations>`_
    - `examples/pyplot <https://github.com/marcomusy/vedo/blob/master/vedo/examples/pyplot>`_
    - `examples/others <https://github.com/marcomusy/vedo/blob/master/vedo/examples/other>`_
    - `examples/others/dolfin <https://github.com/marcomusy/vedo/blob/master/vedo/examples/other/dolfin>`_.
    - `examples/others/trimesh <https://github.com/marcomusy/vedo/blob/master/vedo/examples/other/trimesh>`_
"""
from __future__ import print_function

__author__     = "Marco Musy"
__license__    = "MIT"
__maintainer__ = "M. Musy, G. Dalmasso"
__email__      = "marco.musy@embl.es"
__status__     = "dev"
__website__    = "https://github.com/marcomusy/vedo"

from vedo.version import _version as __version__
from vedo.plotter import *
from vedo.analysis import *
from vedo.shapes import *
from vedo.vtkio import *

from vedo.base import BaseActor, BaseGrid
from vedo.ugrid import UGrid
from vedo.assembly import Assembly
from vedo.mesh import Mesh, merge
from vedo.picture import Picture
from vedo.volume import Volume
from vedo.tetmesh import *

from vedo.utils import *
from vedo.colors import *
import vedo.settings as settings
import vedo.addons as addons
from vedo.settings import datadir, embedWindow

# hack: need to uncomment this to generate documentation html
from vedo.dolfin import _inputsort

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


