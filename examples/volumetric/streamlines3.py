"""Draw streamlines for the cavity case from OpenFOAM tutorial"""
from vedo import *

# Load file as type vtkUnStructuredGrid
fpath = download(dataurl+"cavity.vtk")
ugrid = loadUnStructuredGrid(fpath)

# Make a grid of points to probe as type Mesh
probe = Grid(pos=[0.05,0.08,0.005], normal=[0,1,0], s=[0.1,0.01], res=[20,4], c='k')

# Compute stream lines with Runge-Kutta4, return a Mesh
stream = StreamLines(
    ugrid, probe,
    active_vectors='U', # name of the active array
    #tubes={"radius":1e-04, "vary_radius":2},
    lw=2, # line width
)

# Make a cloud of points form the ugrid, in order to draw arrows
domain = Points(ugrid)
coords = domain.points()
vects  = domain.pointdata['U']/200
arrows = Arrows(coords-vects, coords+vects, c='jet_r') # use colormap
box    = domain.box().c('k')  # build a box frame of the domain

show(stream, arrows, box, probe, __doc__, axes=5).close()
