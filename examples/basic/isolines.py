"""
Draw the isolines of the
active scalars on a surface
"""
from vtkplotter import *

mesh = Hyperboloid().rotateX(20) # a whatever mesh

scals = mesh.getPoints()[:,1] # pick x coords of vertices
mesh.pointColors(scals).addScalarBar(c='w')

isols = mesh.isolines(n=12, vmin=0).color('w')

show(mesh, isols, Text(__doc__), axes=1)

