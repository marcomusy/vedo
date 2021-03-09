"""Smoothing a mesh"""
from vedo import *

s1 = load(dataurl+'panther.stl').lw(0.1)

s2 = s1.clone().x(50).c('lb').lw(0)
s2.subdivide(3).smoothWSinc().computeNormals()
s2.lighting('glossy').phong()

# other useful filters to combine are
# mesh.decimate(), clean(), smoothLaplacian(), smoothMLS2D()

cam = dict(pos=(113, -189, 62.1),
           focalPoint=(18.3, 4.39, 2.41),
           viewup=(-0.0708, 0.263, 0.962),
           distance=223,
           clippingRange=(74.3, 382))

show(s1, s2, __doc__, bg='k', bg2='lg', axes=11, camera=cam)
