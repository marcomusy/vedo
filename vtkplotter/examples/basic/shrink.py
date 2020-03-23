"""
Shrink the polygons of a mesh
to make the inside visible.
"""
from vtkplotter import *

pot = load(datadir+"teapot.vtk").shrink(0.75)

s = Sphere(r=0.2).pos(0, 0, -0.5)

show(pot, s, Text2D(__doc__), viewup="z")
