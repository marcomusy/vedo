"""Hover mouse on mesh to
visualize object details"""
from vedo import *

mesh = Mesh(datadir+"bunny.obj")

# Create multiple arrays associated to mesh vertices or cells
mesh.addPointArray(mesh.points()[:,0], name='MYPOINTARRAY')
mesh.addCellArray(mesh.cellCenters()[:,1], name='MYCELLARRAY')

# Create more objects
sph = Sphere(r=0.02, pos=(-0.1,0.05,0.05))
cub = Cube().alpha(0.5).lineWidth(2)

pts = Points(cub.points(), r=50, c='v')
pts.name = 'The cube vertices'  # can give a name to any objects

# Create an instance of the plotter window
plt = Plotter(N=2, axes=1, sharecam=False)

# Add a 2D text legend to both renderers and show:
plt.addTextLegend(at=0).show(mesh, sph, __doc__, at=0)
plt.addTextLegend(at=1).show(cub, pts, at=1)
interactive()