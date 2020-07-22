"""Density field as a Volume from a point cloud"""

from vedo import *

s  = load(datadir+'bunny.obj').normalize().subdivide(2).pointSize(2)

vol= s.density().printInfo()

plane = probePlane(vol, normal=(1,1,1)).alpha(0.5)

show([("Point cloud", s),
      ("Point density as Volume", vol, vol.box(), plane) ],
     N=2,
     )