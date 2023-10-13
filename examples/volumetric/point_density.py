"""Density field as a Volume from a point cloud"""
from vedo import *

surf = Mesh(dataurl+'bunny.obj').normalize().subdivide(2)
surf.color("k5").point_size(2)

vol = surf.density()

plane = vol.probe_plane(normal=(1,1,1)).alpha(0.5)

show([
      ("Point cloud", surf),
      ("Point density as Volume", vol, vol.box(), plane)
     ], N=2,
).close()

