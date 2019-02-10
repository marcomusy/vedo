"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png

**A python class to easily draw, analyse and animate 3D objects and point clouds with VTK.**

.. note:: **Please check out the** `git repository <https://github.com/marcomusy/vtkplotter>`_.

    A full list of examples can be found in directories:
        
    - `examples/basic <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic>`_ ,
    - `examples/advanced <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced>`_ ,
    - `examples/volumetric <https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric>`_,
    - `examples/simulations <https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations>`_
    - `examples/others <https://github.com/marcomusy/vtkplotter/blob/master/examples/other>`_.



Citations and References
^^^^^^^^^^^^^^^^^^^^^^^^

Publications where `vtkplotter` has been used so far:

1. Diego, X. *et al,*: *"Key features of Turing systems are determined purely by network topology"*, 
`Physical Review X 20 June 2018 <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021071>`_. 

2. M. Musy, K. Flaherty, J. Raspopovic, A. Robert-Moreno, J. T. Richtsmeier, J. Sharpe:
*"A Quantitative Method for Staging Mouse Limb Embryos based on Limb Morphometry"*,
Development 2018, `doi: 10.1242/dev.154856 <http://dev.biologists.org/content/145/7/dev154856>`_, 
5 April 2018.

3. G. Dalmasso *et al.*, "Evolution in space and time of 3D volumetric images", in preparation.

Have you found this useful? Cite it if you wish as:
    
M. Musy, G. Dalmasso, & B. Sullivan. (2019, February 10). 
`vtkplotter (Version v8.9.0). Zenodo. http://doi.org/10.5281/zenodo.2561402`_
"""

__author__     = "Marco Musy"
__license__    = "MIT"
__maintainer__ = "M. Musy, G. Dalmasso"
__email__      = "marco.musy@embl.es"
__status__     = "dev"
__website__    = "https://github.com/marcomusy/vtkplotter"
__version__    = "8.9.0" ### defined also in setup.py and docs/source/conf.py

from vtkplotter.plotter import *
from vtkplotter.analysis import *
from vtkplotter.shapes import *
from vtkplotter.vtkio import *
from vtkplotter.actors import *
from vtkplotter.utils import *
from vtkplotter.colors import *
import vtkplotter.settings as settings

from numpy import sin, cos, sqrt, exp, log, dot, cross, array, arange

# imports hierarchy
# plotter : utils, colors, actors, vtkio, shapes
# analysis: utils, colors, actors, vtkio, shapes
# shapes  : utils, colors, actors
# vtkio   : utils, colors, actors
# actors  : utils, colors
# utils   :        colors
# colors  : -


def _ignore_warnings():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
_ignore_warnings()

###############
settings.init()
###############




# deprecated stuff
def _deprecated_msg(cls):
	print('\nDeprecated in version > 8.9:',
       cls+'(). Use '+ cls.capitalize()+ '() with capital letter instead. Abort.')
	exit()

def point(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('point')
def points(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('points')
def glyph(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('glyph')
def line(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('line')
def tube(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('tube')
def lines(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('lines')
def ribbon(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('ribbon')
def arrow(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('arrow')
def arrows(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('arrows')
def polygon(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('polygon')
def rectangle(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('rectangle')
def disc(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('disc')
def sphere(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('sphere')
def spheres(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('spheres')
def earth(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('earth')
def ellipsoid(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('ellipsoid')
def grid(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('grid')
def plane(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('plane')
def box(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('box')
def cube(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('cube')
def helix(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('helix')
def cylinder(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('cylinder')
def cone(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('cone')
def pyramid(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('pyramid')
def torus(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('torus')
def paraboloid(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('paraboloid')
def hyperboloid(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('hyperboloid')
def text(*args, **kwargs):
    '''Deprecated. Use capital letter.'''
    _deprecated_msg('text')
