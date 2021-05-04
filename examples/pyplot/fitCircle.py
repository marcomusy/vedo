"""Fit circles analytically to measure
the signed curvature of a line"""
from vedo import *

shape = Spline([[1.0, 2.0, -1.],
                [1.5, 0.0, 0.4],
                [2.0, 4.0, 0.5],
                [4.0, 1.5, -.3]], res=200)

points = shape.points()
fitpts, circles, curvs = [], [], []
n = 3                   # nr. of points to use for the fit
for i in range(shape.NPoints() - n):
    pts = points[i:i+n]
    center, R, normal = fitCircle(pts)
    z = cross(pts[-1]-pts[0], center-pts[0])[2]
    curvs.append(sqrt(1/R)*z/abs(z))
    if R < 0.75:
        circle = Circle(center, r=R).wireframe().orientation(normal)
        circles.append(circle)
        fitpts.append(center)
curvs += [curvs[-1]]*n  # fill the missing last n points

shape.lw(8).cmap('rainbow', curvs).addScalarBar3D(title='\pm1/\sqrtR')
show(shape, circles, Points(fitpts), __doc__, axes=1).close()
