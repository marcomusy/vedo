"""Extract points on the boundary of a mesh.
Add an ID label to all vertices."""
from vedo import *

b = Mesh(dataurl+'290.vtk')
b.computeNormals().clean().lw(0.1)

pids = b.boundaries(returnPointIds=True)
bpts = b.points()

pts = Points(bpts[pids], r=10, c='red5')

labels = b.labels('id', scale=10).c('green2')

show(b, pts, labels, __doc__, zoom=2).close()
