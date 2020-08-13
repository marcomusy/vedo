"""Generate the silhouette of a mesh
as seen along a specified direction
"""
from vedo import *


plt = Plotter(title="Example of `projectOnPlane`.")

s = Hyperboloid().rotateX(20)
pts = s.points()

# orthogonal projections to x-plane, y-plane and z-plane
sx = s.clone().projectOnPlane('x').c('r').x(-3) # sx is 2d
sy = s.clone().projectOnPlane('y').c('g').y(-3)
sz = s.clone().projectOnPlane('z').c('b').z(-6)
plt += s
plt += sx
plt += sx.silhouette('2d')
plt += sy
plt += sy.silhouette('2d')
plt += sz
plt += sz.silhouette('2d')

# orthogonal projection
plane1 = Plane(pos=(2, 0, 2), normal=(1, 0, 1), sx=5).alpha(0.1)
so = s.clone().projectOnPlane(plane1).c('y')
plt += plane1
plt += so
plt += so.silhouette('2d')
pts1 = so.silhouette('2d').points()
for i in range(0, len(pts), int(len(pts) / 16)):
     plt += Line(pts1[i], pts[i], c='k', alpha=0.3, lw=0.5)

# perspective projection
plane2 = Plane(pos=(3, 3, 3), normal=(1, 1, 1), sx=5).alpha(0.1)
point = [6, 6, 6]
sp = s.clone().projectOnPlane(plane2, point=point).c('m')
for i in range(0, len(pts), int(len(pts) / 16)):
     plt += Line(point, pts[i], c='k', alpha=0.3, lw=0.5)
plt += plane2
plt += sp
plt += sp.silhouette('2d')

# oblique projection
plane3 = Plane(pos=(4, 8, -4), normal=(-1, 0, 1), sx=5).alpha(0.1)
sob = s.clone().projectOnPlane(plane3, direction=(1, 2, -1)).c('g')
plt += plane3
plt += sob
plt += sob.silhouette('2d')
pts2 = sob.silhouette('2d').points()
for i in range(0, len(pts), int(len(pts) / 16)):
     plt += Line(pts2[i], pts[i], c='k', alpha=0.3, lw=0.5)

plt.show(viewup="z", axes={'zxGrid':True}, interactive=1)
