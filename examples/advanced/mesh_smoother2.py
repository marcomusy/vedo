"""Smoothing a mesh"""
from vedo import dataurl, Mesh, show

s1 = Mesh(dataurl+'panther.stl').lw(0.1)

s2 = s1.clone().x(50)  # place at x=50
s2.subdivide(3).smooth().computeNormals()
s2.c('light blue').lw(0).lighting('glossy').phong()

# other useful filters to combine are
# mesh.decimate(), clean(), smooth()

cam = dict(pos=(113, -189, 62.1),
           focalPoint=(18.3, 4.39, 2.41),
           viewup=(-0.0708, 0.263, 0.962),
           distance=223,
)
show(s1, s2, __doc__,
     bg='black', bg2='lightgreen', axes=11, camera=cam).close()
