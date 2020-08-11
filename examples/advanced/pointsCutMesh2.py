"""Create a closed spline from a bunch of
random points on a sphere then create a ribbon
surface and cut through the sphere with that"""
from vedo import *

s = Sphere().computeNormals().alpha(0.2).lw(0.1)

sc = s.points()
sn = s.normals(cells=True) # cell normals

# pick a few points
pts = (sc[10], sc[15], sc[129], sc[165], sc[10])

#build the spline
ptsm = Spline(pts, smooth=0).c('k').lw(4)
ptsmc = ptsm.points()

# and a ribbon-like surface using cell normals
splinen = []
for p in ptsmc:
    iclos = s.closestPoint(p, returnIds=True)
    splinen.append(sn[iclos])
pts0 = ptsmc - 0.1*vector(splinen)
pts1 = ptsmc + 0.2*vector(splinen)
rb = Ribbon(pts0, pts1).bc('green')

#cut with the ribbon and then with a yz plane
scut = s.clone().c('blue').alpha(0.7).cutWithMesh(rb)

show(s, Points(pts), ptsm, rb, scut, __doc__, axes=1)
