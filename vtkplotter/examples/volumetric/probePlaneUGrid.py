"""Probe a vtkUnStructuredGrid with a plane"""
from vtkplotter import *

# same could be done with vtkRectilinearGrid etc..
data = loadUnStructuredGrid(datadir+"ugrid.vtk")

# create the outline of the data
outermesh = Mesh(data).alpha(0.5).wireframe()
orig = data.GetCenter()

pl = probePlane(data, origin=orig, normal=(0.1,0.2,1))

#pl.printInfo()
#pl.pointColors('scalars', cmap='hot')

show(pl, outermesh, axes=1)
