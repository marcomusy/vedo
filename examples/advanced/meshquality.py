"""
Calculate functions of quality of the elements of a triangular mesh.
"""
print(__doc__)

from vtkplotter import *

a1 = meshQuality(Sphere())
a2 = meshQuality(load(datadir+"shapes/bunny.obj").normalize())
a3 = meshQuality(load(datadir+"shapes/motor.g").normalize())

show([a1, a2, a3], N=3)
