"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png

A python module for scientific visualization,  
analysis and animation of 3D objects and point clouds based on VTK.

.. note:: **Please check out the** `git repository <https://github.com/marcomusy/vtkplotter>`_.

    A full list of examples can be found in directories:
        
    - `examples/basic <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic>`_ ,
    - `examples/advanced <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced>`_ ,
    - `examples/volumetric <https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric>`_,
    - `examples/simulations <https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations>`_
    - `examples/others <https://github.com/marcomusy/vtkplotter/blob/master/examples/other>`_.



References
^^^^^^^^^^^

Publications where ``vtkplotter`` has been used so far:

1. Diego, X. *et al,*: *"Key features of Turing systems are determined purely by network topology"*, 
`Physical Review X 20 June 2018 <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021071>`_. 

2. M. Musy, K. Flaherty, J. Raspopovic, A. Robert-Moreno, J. T. Richtsmeier, J. Sharpe:
*"A Quantitative Method for Staging Mouse Limb Embryos based on Limb Morphometry"*,
Development 2018, `doi: 10.1242/dev.154856 <http://dev.biologists.org/content/145/7/dev154856>`_, 
5 April 2018.

3. G. Dalmasso *et al.*, "Evolution in space and time of 3D volumetric images", in preparation.

**Have you found this software useful for your research? Please cite it as:**
    
M. Musy, G. Dalmasso, & B. Sullivan.  
"`vtkplotter`, a python module for scientific visualization,  
analysis and animation of 3D objects and point clouds based on VTK." (version v8.9.0). Zenodo, 
`doi: 10.5281/zenodo.2561402 <http://doi.org/10.5281/zenodo.2561402>`_, 10 February 2019.
"""
from __future__ import print_function

__author__ = "Marco Musy"
__license__ = "MIT"
__maintainer__ = "M. Musy, G. Dalmasso"
__email__ = "marco.musy@embl.es"
__status__ = "dev"
__website__ = "https://github.com/marcomusy/vtkplotter"
__version__ = "2019.1.1"  ### defined also above, in setup.py and docs/source/conf.py

from vtkplotter.plotter import *
from vtkplotter.analysis import *
from vtkplotter.shapes import *
from vtkplotter.vtkio import *
from vtkplotter.actors import *
from vtkplotter.utils import *
from vtkplotter.colors import *
import vtkplotter.settings as settings
from vtkplotter.settings import datadir
import vtkplotter.dolfin as dolfin
from numpy import sin, cos, sqrt, exp, log, dot, cross, array, arange

# imports hierarchy
# plotter : utils, colors, actors, vtkio, shapes
# analysis: utils, colors, actors, vtkio, shapes
# shapes  : utils, colors, actors
# vtkio   : utils, colors, actors
# actors  : utils, colors
# utils   :        colors
# colors  : -


###############
settings._init()
###############


#####################
def _deprecated_msg(cls):
	print('\nDeprecated in version > 8.9:',
              cls+'(). Use '+ cls.capitalize()+ '() with capital letter instead.\n')

def point(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("point")
    return Point(*args, **kwargs)


def points(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("points")
    return Points(*args, **kwargs)


def glyph(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("glyph")
    return Glyph(*args, **kwargs)


def line(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("line")
    return Line(*args, **kwargs)


def tube(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("tube")
    return Tube(*args, **kwargs)


def lines(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("lines")
    return Lines(*args, **kwargs)


def ribbon(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("ribbon")
    return Ribbon(*args, **kwargs)


def arrow(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("arrow")
    return Arrow(*args, **kwargs)


def arrows(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("arrows")
    return Arrows(*args, **kwargs)


def polygon(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("polygon")
    return Polygon(*args, **kwargs)


def rectangle(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("rectangle")
    return Rectangle(*args, **kwargs)


def disc(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("disc")
    return Disc(*args, **kwargs)


def sphere(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("sphere")
    return Sphere(*args, **kwargs)


def spheres(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("spheres")
    return Spheres(*args, **kwargs)


def earth(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("earth")
    return Earth(*args, **kwargs)


def ellipsoid(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("ellipsoid")
    return Ellipsoid(*args, **kwargs)


def grid(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("grid")
    return Grid(*args, **kwargs)


def plane(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("plane")
    return Plane(*args, **kwargs)


def box(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("box")
    return Box(*args, **kwargs)


def cube(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("cube")
    return Cube(*args, **kwargs)


def helix(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("helix")
    return Helix(*args, **kwargs)


def cylinder(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("cylinder")
    return Cylinder(*args, **kwargs)


def cone(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("cone")
    return Cone(*args, **kwargs)


def pyramid(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("pyramid")
    return Pyramid(*args, **kwargs)


def torus(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("torus")
    return Pyramid(*args, **kwargs)


def paraboloid(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("paraboloid")
    return Paraboloid(*args, **kwargs)


def hyperboloid(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("hyperboloid")
    return Hyperboloid(*args, **kwargs)


def text(*args, **kwargs):
    """Deprecated. Use capital letter."""
    _deprecated_msg("text")
    return Text(*args, **kwargs)
