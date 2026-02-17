"""Intersection of two polygonal meshes"""
from vedo import *

settings.use_depth_peeling = True

# Semi-transparent car surface.
car = Mesh(dataurl+"porsche.ply").alpha(0.2)

# Tube-like surface intersecting the car.
line = [(-9.,0.,0.), (0.,1.,0.), (9.,0.,0.)]
tube = Tube(line).triangulate().c("violet",0.2)

# Intersection contour between the two surfaces.
contour = car.intersect_with(tube).linewidth(4).c('black')

show(car, tube, contour, __doc__, axes=7).close()
