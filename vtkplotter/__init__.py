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

M. Musy, et al.,
"`vtkplotter`, a python module for scientific visualization,
analysis and animation of 3D objects and point clouds based on VTK." (version v8.9.0). Zenodo,
`doi: 10.5281/zenodo.2561402 <http://doi.org/10.5281/zenodo.2561402>`_, 10 February 2019.
"""
from __future__ import print_function
from vtkplotter.version import _version

__author__ = "Marco Musy"
__license__ = "MIT"
__maintainer__ = "M. Musy, G. Dalmasso"
__email__ = "marco.musy@embl.es"
__status__ = "dev"
__website__ = "https://github.com/marcomusy/vtkplotter"
__version__ = _version

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


## deprecations
def loadImageData(*args, **kargs):
    "Do not use."
    printc("~bomb loadImageData has been retired in version>=3.0. Use instead:", c=1)
    printc("img = load('file.tif').imagedata() # or better:")
    printc("vol = load('file.tif') # returns a Volume")
    printc("Abort.", c=1)
    exit(0)






