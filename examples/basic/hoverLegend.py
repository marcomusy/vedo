"""Hover mouse on mesh to
visualize object details"""
from vedo import *

mesh = Mesh(dataurl+"bunny.obj").color('k7')

# Create multiple arrays associated to mesh vertices or cells
mesh.pointdata['MYPOINTARRAY'] = mesh.points()[:,0]
mesh.celldata['MYCELLARRAY']   = mesh.cellCenters()[:,1]

# Create more objects
sph = Sphere(r=0.02, pos=(-0.1,0.05,0.05))
cub = Cube().alpha(0.5).lineWidth(2)

pts = Points(cub.points(), r=50, c='v')
pts.name = 'The cube vertices'  # can give a name to any objects

# Create an instance of the plotter window
plt = Plotter(N=2, axes=1, sharecam=False)

# Add a 2D hover legend to both renderers and show:
plt.at(0).addHoverLegend().show(mesh, sph, __doc__)
plt.at(1).addHoverLegend().show(cub, pts)
plt.interactive().close()
