"""Smoothing a mesh"""
from vedo import *

s1 = load(datadir+'panther.stl').lw(0.1)

s2 = s1.clone().c('lb').lw(0)
s2.subdivide(3).smoothWSinc().computeNormals()
s2.lighting('glossy').phong()

# other useful filters to combine are
# mesh.decimate(), clean(), smoothLaplacian(), smoothMLS2D()

show([(s1, __doc__), s2],
     N=2, viewup='z', bg='k', bg2='lg', axes=11)
