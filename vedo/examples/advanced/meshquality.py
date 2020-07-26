"""Metrics of quality for
the cells of a triangular mesh
(zoom to see cell label values)
"""
from vedo import *
from vedo.pyplot import histogram

mesh = load(datadir+"bunny.obj").scale(100).computeNormals()
# mesh = load(datadir+"panther.stl").scale(10).computeNormals()
mesh.lineWidth(0.1)

# generate an array for mesh quality
arr = mesh.quality(cmap='RdYlBu')

hist = histogram(arr, xtitle='mesh quality', bc='w')
# make it smaller and position it
hist.rotateX(-90).scale(.1).pos(2,3.5,10)

# add a scalar bar for the active scalars
mesh.addScalarBar()

# create numeric labels of active scalar on top of cells
labs = mesh.labels(cells=True,
                   precision=3,
                   font='Quikhand',
                   c='black',
                   ratio=5)

show(mesh, labs, hist, __doc__, bg='bb')
