"""Hover mouse on mesh to
visualize object details"""
from vedo import *

mesh = Mesh(dataurl+"bunny.obj").color('k7')

# Create multiple arrays associated to mesh vertices or cells
mesh.pointdata['MYPOINTARRAY'] = mesh.vertices[:,0]
mesh.celldata['MYCELLARRAY']   = mesh.cell_centers[:,1]

# Create more objects
sph = Sphere(pos=(-0.1,0.05,0.05), r=0.02)
cub = Cube().alpha(0.5).linewidth(2)

pts = Points(cub).c("violet").point_size(50)
pts.name = 'The cube vertices'  # can give a name to any objects

# Create an instance of the plotter window
plt = Plotter(N=2, sharecam=False)

# Add a 2D hover legend to both renderers and show:
cid0 = plt.at(0).add_hover_legend()
plt.show(mesh, sph, __doc__)

cid1 = plt.at(1).add_hover_legend()
plt.show(cub, pts)

plt.interactive().close()

