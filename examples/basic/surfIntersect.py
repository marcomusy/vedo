"""
1D intersection of two polygonal meshes
"""
from vtkplotter import *

car = load(datadir+"porsche.ply").alpha(0.2)

s = Tube([(-9.,0.,0.), (0.,1.,0.), (9.,0.,0.)])
s.triangle().clean().color("violet").alpha(0.2)

contour = surfaceIntersection(car, s)
contour.lw(4).printInfo()

show(s, car, contour, Text(__doc__), bg='w')
