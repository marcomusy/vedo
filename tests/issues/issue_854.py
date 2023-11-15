from vedo import *

msh = ParametricShape("RandomHills").scale(2)

spline = Spline([[1,1,-1], [0,2,0], [1,3,3]]).lw(3)

pts = spline.vertices
cpts = []
for i in range(spline.npoints-1):
    p = pts[i]
    q = pts[i+1]
    ipts = msh.intersect_with_line(p, q)
    if len(ipts):
        cpts.append(ipts[0])

cpts = Points(cpts, r=12)

show(msh, spline, cpts, axes=1, viewup="z")
