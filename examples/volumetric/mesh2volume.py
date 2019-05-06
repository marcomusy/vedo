"""
Convert a mesh it into volume (left in grey) where the
foreground voxels are 1 and the background voxels are 0.

Right: the Volume is isosurfaced.
"""
from vtkplotter import *

doc = Text(__doc__, c="k")

s = load(datadir+"bunny.obj").normalize().wire()

v = actor2Volume(s, spacing=(0.02, 0.02, 0.02)).alpha([0,0.5]).c('blue')

iso = isosurface(v, smoothing=0.9).color("b")

show(v, s.scale(1.05), doc, at=0, N=2, bg="w")

show(iso, at=1, interactive=1)
