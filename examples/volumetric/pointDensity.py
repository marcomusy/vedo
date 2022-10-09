"""Density field as a Volume from a point cloud"""
from vedo import *

s = Mesh(dataurl+'bunny.obj').normalize().subdivide(2).point_size(3).c("black")

vol = s.density().print()

plane = probe_plane(vol, normal=(1,1,1)).alpha(0.5)

show([
      ("Point cloud", s),
      ("Point density as Volume", vol, vol.box(), plane)
     ], N=2,
).close()

