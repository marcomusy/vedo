"""Read and show meshio objects"""
import meshio
from vtkplotter import datadir, show, Mesh

mesh = meshio.read(datadir+'shuttle.obj')

# vtkplotter understands meshio format for polygonal data:
#show(mesh, __doc__)

# explicitly convert it to a vtkplotter.Mesh object:
m = Mesh(mesh).lineWidth(1).color('tomato').printInfo()
show(m, __doc__)
