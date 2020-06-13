"""Extract points on the boundary of a mesh.
Add a label to all vertices. """
from vedo import *

b = load(datadir+'290.vtk')
b.computeNormals().clean().lw(0.1)

pids = b.boundaries(returnPointIds=True)
bpts = b.points()[pids]

pts = Points(bpts, r=10, c='red')

labs = b.labels('id').c('dg') # add point labels

show(b, pts, labs, __doc__, zoom=2)