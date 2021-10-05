"""Set a loop of random points on a sphere
to cut a region of the mesh"""
from vedo import *

settings.useDepthPeeling = True

s = Sphere().alpha(0.2).lw(0.1)

# pick a few points on the sphere
sc = s.points()
pts = Points([sc[10], sc[15], sc[129], sc[165]], r=12)

#cut loop region identified by the points
scut = s.clone().cutWithPointLoop(pts, invert=False)
scut.c('blue',0.7).lw(0).scale(1.03)

show(s, pts, scut, __doc__, axes=1)
