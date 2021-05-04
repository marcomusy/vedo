"""Find the intersection points of two coplanar lines"""
import numpy as np
from vedo import *

p1, p2 = (-1,-1,0), (10,2,0)

x = np.linspace(0,10, 50)
y = np.sin(x)*4
pts = np.c_[x,y]

# create 2 lines and assign some arbitrary rotations
line1 = Spline(pts).lw(5).c('black').rotateY(10).rotateX(15)
line2 = Line(p1,p2).lw(5).c('green').rotateY(10).rotateX(15)

# make a small extrusion of line1 and intersect it with line2:
ds = line1.diagonalSize()*0.02  # 1% tolerance
pint = line1.extrude(ds).shift(0,0,-ds/2).intersectWithLine(line2)
ps = Points(pint, r=15).c('red')

# lets fill the convex area between the first 2 hits:
id0 = line1.closestPoint(pint[0], returnPointId=True)
id1 = line1.closestPoint(pint[1], returnPointId=True)
msh = Line(line1.points()[id0:id1]).triangulate().lw(0).shift(0,0,-0.01)

show(line1, line2, ps, msh, __doc__+f"\narea = {msh.area()} cm\^2", axes=1).close()
