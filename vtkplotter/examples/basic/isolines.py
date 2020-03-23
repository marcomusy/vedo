"""
Draw the isolines of a
scalar field on a surface
"""
from vtkplotter import *

mesh = ParametricShape('RandomHills') # a whatever mesh

pts = mesh.points() 
# use z coords of vertices as scalars:
mesh.pointColors(pts[:,2], cmap='terrain').addScalarBar()

isols = mesh.isolines(n=10, vmin=-0.1).color('w')

show(mesh, isols, Text2D(__doc__), axes=1, viewup='z')
