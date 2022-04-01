"""Metrics of quality for
the cells of a triangular mesh
(zoom to see cell label values)"""
from vedo import dataurl, Mesh, show
from vedo.pyplot import histogram

mesh = Mesh(dataurl+"panther.stl").computeNormals().lineWidth(0.1).flat()

# generate a numpy array for mesh quality
mesh.addQuality(measure=6).cmap('RdYlBu', on='cells').print()

hist = histogram(mesh.celldata["Quality"], xtitle='mesh quality', ac='w')
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

cam = dict(pos=(59.8, -191, 78.9),
           focalPoint=(27.9, -2.94, 3.33),
           viewup=(-0.0170, 0.370, 0.929),
           distance=205,
           clippingRange=(87.8, 355))

show(mesh, labs, hist, __doc__, bg='bb', camera=cam, axes=11).close()
