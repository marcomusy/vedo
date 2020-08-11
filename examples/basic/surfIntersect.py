"""Intersection of two polygonal meshes"""
from vedo import *

car = load(datadir+"porsche.ply").alpha(0.2)

line = [(-9.,0.,0.), (0.,1.,0.), (9.,0.,0.)]
tube = Tube(line).triangulate().c("violet").alpha(0.2)

contour = car.intersectWith(tube).lineWidth(4).c('black')

show(car, tube, contour, __doc__, axes=7)
