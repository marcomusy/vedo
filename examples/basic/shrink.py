"""Shrink mesh polygons
to make the inside visible"""
from vedo import *

# Shrink each polygon towards its center.
pot = Mesh(dataurl+"teapot.vtk").shrink(0.75)
# Add a reference sphere inside the teapot.
s = Sphere(r=0.2).pos(0, 0, -0.5)

show(pot, s, __doc__, axes=11, viewup="z").close()
