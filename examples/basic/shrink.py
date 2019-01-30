'''
Shrink the triangulation of a mesh to make the inside visible.
'''
from vtkplotter import load, sphere, show, text

pot = load('data/shapes/teapot.vtk').shrink(0.75)

s = sphere(r=0.2).pos(0,0,-0.5)

show([pot, s, text(__doc__)], viewup='z')
