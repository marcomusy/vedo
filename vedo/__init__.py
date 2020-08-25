"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png

A python module for scientific visualization,
analysis and animation of 3D objects and point clouds based on VTK.

.. note:: **Please check out the** `git repository <https://github.com/marcomusy/vedo>`_.

    A full list of examples can be found in directories:

    - `examples/basic <https://github.com/marcomusy/vedo/tree/master/examples/basic>`_
    - `examples/advanced <https://github.com/marcomusy/vedo/tree/master/examples/advanced>`_
    - `examples/volumetric <https://github.com/marcomusy/vedo/tree/master/examples/volumetric>`_
    - `examples/tetmesh <https://github.com/marcomusy/vedo/tree/master/examples/tetmesh>`_
    - `examples/simulations <https://github.com/marcomusy/vedo/tree/master/examples/simulations>`_
    - `examples/pyplot <https://github.com/marcomusy/vedo/tree/master/examples/pyplot>`_
    - `examples/others <https://github.com/marcomusy/vedo/tree/master/examples/other>`_
    - `examples/others/dolfin <https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin>`_
    - `examples/others/trimesh <https://github.com/marcomusy/vedo/tree/master/examples/other/trimesh>`_
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
from vedo.shapes import *
from vedo.io import *

from vedo.base import *
from vedo.ugrid import UGrid
from vedo.assembly import Assembly, procrustesAlignment
from vedo.pointcloud import *
from vedo.mesh import *
from vedo.picture import Picture
from vedo.volume import *
from vedo.tetmesh import *

from vedo.utils import *
from vedo.colors import *
import vedo.settings as settings
import vedo.addons as addons
from vedo.addons import Ruler, Goniometer, buildRulerAxes, buildAxes
from vedo.settings import datadir, embedWindow

# hack: need to uncomment this to generate documentation html
from vedo.dolfin import _inputsort

from numpy import sin, cos, sqrt, exp, log, dot, cross, array, arange


###########################################################################
settings._init()
###########################################################################


## deprecations ############################################################
def alignICP(source, target, iters=100, rigid=False, invert=False, useCentroids=False):
    """Obsolete: Please use source.alignTo(target) instead."""
    printc("Obsolete alignICP: Please use source.alignTo(target) instead",
                  c=1, box='-')
    return source.alignTo(target, iters, rigid, invert, useCentroids)

def booleanOperation(mesh1, operation, mesh2):
    """Obsolete: Please use mesh1.boolean(operation, mesh2) instead."""
    printc("Obsolete booleanOperation: Please use mesh1.boolean(operation, mesh2) instead",
                  c=1, box='-')
    return mesh1.boolean(operation, mesh2)

def surfaceIntersection(mesh1, mesh2, tol=1e-06):
    """Obsolete: Please use mesh1.intersectWith(mesh2) instead."""
    printc("Obsolete surfaceIntersection: Please use mesh1.intersectWith(mesh2) instead",
                  c=1, box='-')
    return mesh1.intersectWith(mesh2)


