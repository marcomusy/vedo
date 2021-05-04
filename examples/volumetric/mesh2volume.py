"""Left: build a volume (grey) from a mesh where the
foreground voxels are 1 and the background voxels are 0.

Right: the Volume is isosurfaced."""
from vedo import *

s = Mesh(dataurl+"bunny.obj").normalize().wireframe()

v = mesh2Volume(s, spacing=(0.02, 0.02, 0.02)).alpha([0,0.5]).c('blue')

iso = v.isosurface().color("b")

show(v, s.scale(1.05), __doc__, at=0, N=2)

show(iso, at=1, interactive=1).close()
