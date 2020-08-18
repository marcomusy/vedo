"""Smoothing a mesh"""
from vedo import *

s1 = load(datadir+'panther.stl').lw(0.1)

s2 = s1.clone().c('lb').lw(0)
s2.subdivide(3).smoothWSinc().computeNormals()
s2.lighting('glossy').phong()

# other useful filters to combine are
# mesh.decimate(), clean(), smoothLaplacian(), smoothMLS2D()

cam = dict(pos=(74.9, -118, 42.7),
           focalPoint=(-5.86, -8.25, 8.01),
           viewup=(-0.0442, 0.272, 0.961),
           distance=140,
           clippingRange=(32.6, 273))

show([(s1, __doc__), s2],
     N=2, bg='k', bg2='lg', axes=11, camera=cam, resetcam=0)
