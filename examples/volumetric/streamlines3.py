"""Draw streamlines for the cavity case from OpenFOAM tutorial"""
from vedo import *

# Load an UnStructuredGrid
ugrid = UnstructuredGrid(dataurl+"cavity.vtk").alpha(0.1)

# Make a grid of points to probe as type Mesh
probe = Grid(s=[0.1,0.01], res=[20,4], c='k')
probe.rotate_x(90).pos(0.05,0.08,0.005)

# Compute stream lines with Runge-Kutta4, return a Mesh
ugrid.pointdata.select('U') # select active vector
print(ugrid)

coords = ugrid.vertices
vects  = ugrid.pointdata['U']/200
arrows = Arrows(coords-vects, coords+vects, c='jet_r') # use colormap

stream = ugrid.compute_streamlines(probe)

show(stream, arrows, ugrid, probe, __doc__, axes=5).close()
