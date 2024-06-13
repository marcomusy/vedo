from vedo import *

aline = Line(Circle().coordinates)
spline = Spline([(0,0,0), (1,1,1), (2,3,3), (1,1,4), (0,1,5)])
spline.lw(5)
pts = spline.coordinates

surfs = []
for i in range(1, len(pts)):
    p0, p1 = pts[i-1:i+1]
    surf = aline.sweep(p1 - p0)
    surfs.append(surf)
surface = merge(surfs, flag=True)
surface.c("gold").lw(0.1).pickable(True)

show(spline, surface, aline, axes=1).close()