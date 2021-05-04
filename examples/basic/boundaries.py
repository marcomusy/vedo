"""Extract points on the boundary of a mesh.
Add an ID label to all vertices."""
from vedo import *

b = Mesh(dataurl+'290.vtk')
b.computeNormals().clean().lw(0.1)

pids = b.boundaries(returnPointIds=True)
bpts = b.points()[pids]

pts = Points(bpts, r=10, c='red')

labels = b.labels('id', scale=10).c('dg')

show(b, pts, labels, __doc__, zoom=2).close()