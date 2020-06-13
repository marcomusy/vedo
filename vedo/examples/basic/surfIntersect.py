"""
1D intersection of two polygonal meshes
"""
from vedo import *

car = load(datadir+"porsche.ply").alpha(0.2)

cline = [(-9.,0.,0.), (0.,1.,0.), (9.,0.,0.)]
t = Tube(cline).triangulate().color("violet").alpha(0.2)

contour = surfaceIntersection(car, t)
contour.lw(4).printInfo()

show(car, t, contour, __doc__)
