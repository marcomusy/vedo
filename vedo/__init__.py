"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png

A python module for scientific visualization,
analysis and animation of 3D objects and point clouds based on VTK and numpy.
"""
__author__     = "Marco Musy"
__license__    = "MIT"
__maintainer__ = "M. Musy"
__email__      = "marco.musy@embl.es"
__status__     = "dev"
__website__    = "https://github.com/marcomusy/vedo"

from vedo.version import _version as __version__
from vedo.plotter import *
from vedo.shapes import *
from vedo.io import *
from vedo.cli import execute_cli

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
import vedo.addons as addons
import vedo.base as base
import vedo.shapes as shapes
from vedo.addons import Ruler, Goniometer, buildRulerAxes, Axes, Light, LegendBox

import vedo.settings as settings
from vedo.settings import dataurl, embedWindow

# hack: need to uncomment this to generate dolfin documentation html
from vedo.dolfin import _inputsort

import vedo.docs as docs  # needed by spyder console, otherwise complains

from numpy import sin, cos, sqrt, exp, log, dot, cross, array

###########################################################################
settings._init()
###########################################################################

