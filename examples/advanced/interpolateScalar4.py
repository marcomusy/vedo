"""Interpolate cell values from a quad-mesh to a tri-mesh"""
from vedo import Grid, show

# Make up some quad mesh with associated scalars
g1 = Grid(res=(25,25)).wireframe(0).lw(1)
scalars = g1.points()[:,1]
g1.cmap("viridis", scalars, vmin=-1, vmax=1, name='gene')
g1.mapPointsToCells() # move the array to cells (faces)
g1.addScalarBar(horizontal=1, pos=(0.7,0.04))
g1.rotateZ(20)  # let's rotate it a bit so it's visible

# Interpolate first mesh onto a new triangular mesh
eps = 0.01
g2 = Grid(res=(50,50)).pos(0.2, 0.2, 0.1).wireframe(0).lw(0)
g2.triangulate()

# Interpolate by averaging the closest 3 points:
#g2.interpolateDataFrom(g1, on='cells', N=3)

# Interpolate by picking points in a specified radius,
#  if there are no points in that radius set null value -1
g2.interpolateDataFrom(
    g1,
    on='cells',
	radius=0.1+eps,
    nullStrategy=1,
    nullValue=-1,
)

g2.cmap('hot', 'gene', on='cells', vmin=-1, vmax=1).addScalarBar()

show(g1, g2, __doc__, axes=1)
