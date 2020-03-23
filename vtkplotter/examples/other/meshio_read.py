"""Read and show meshio objects"""
import meshio
from vtkplotter import datadir, show, Mesh, Text, printc

mesh = meshio.read(datadir+'shuttle.obj')

# vtkplotter understands meshio format:
printc(mesh, c='y')
show(mesh, Text2D(__doc__))

# or explicitly convert it to an Mesh object:
m = Mesh(mesh).lineWidth(1).color('tomato')
show(m)
