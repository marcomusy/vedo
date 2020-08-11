"""Metrics of quality for
the cells of a triangular mesh
(zoom to see cell label values)"""
from vedo import *
from vedo.pyplot import histogram

mesh = load(datadir+"panther.stl").computeNormals().lineWidth(0.1)

# generate a numpy array for mesh quality
arr = mesh.quality(cmap='RdYlBu', measure=6)

hist = histogram(arr, xtitle='mesh quality', bc='w')
# make it smaller and position it, useBounds makes the cam
# ignore the object when resetting the 3d qscene
hist.scale(0.6).pos(40,-53,0).useBounds(False)

# add a scalar bar for the active scalars
mesh.addScalarBar3D(c='w', title='triangle quality by min(\alpha_i )')

# create numeric labels of active scalar on top of cells
labs = mesh.labels(cells=True,
                   precision=3,
                   scale=0.4,
                   font='Quikhand',
                   c='black',
                  )

show(mesh, labs, hist, __doc__, bg='bb', viewup='z')
