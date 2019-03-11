"""
Shrink the triangulation of a mesh 
to make the inside visible.
"""
from vtkplotter import load, Sphere, show, Text

pot = load("data/shapes/teapot.vtk").shrink(0.75)

s = Sphere(r=0.2).pos(0, 0, -0.5)

show([pot, s, Text(__doc__)], viewup="z")
