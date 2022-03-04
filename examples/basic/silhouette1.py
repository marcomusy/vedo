"""Generate the silhouette of a mesh
as seen along a specified direction
"""
from vedo import *

s = Hyperboloid().rotateX(20)

sx = s.clone().projectOnPlane('x').c('r').x(-3) # sx is 2d
sy = s.clone().projectOnPlane('y').c('g').y(-3)
sz = s.clone().projectOnPlane('z').c('b').z(-3)

show(s,
     sx, sx.silhouette('2d'), # 2d objects dont need a direction
     sy, sy.silhouette('2d'),
     sz, sz.silhouette('2d'),
     __doc__,
     axes=7,
     viewup='z',
).close()
