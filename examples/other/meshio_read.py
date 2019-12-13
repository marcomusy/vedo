"""Read and show meshio objects"""
import meshio
from vtkplotter import datadir, show, Actor, Text, printc

mesh = meshio.read(datadir+'shuttle.obj')

# vtkplotter understands meshio format:
printc(mesh, c='y')
show(mesh, Text(__doc__))

# or explicitly convert it to an Actor object:
a = Actor(mesh).lineWidth(1).color('tomato')
show(a)
