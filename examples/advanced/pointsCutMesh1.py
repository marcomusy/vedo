"""Set a loop of random points on a sphere
to cut a region of the mesh.
"""
from vtkplotter import *

s = Sphere().alpha(0.2).lw(0.1)

# pick a few points on the sphere
sc = s.getPoints()
pts = Points([sc[10], sc[15], sc[129], sc[165]])

#cut loop region identified by the points
scut = s.clone().cutWithPointLoop(pts, invert=False)
scut.c('blue').alpha(0.7).scale(1.05)

show(s, pts, scut, Text(__doc__), axes=1, bg='w')

