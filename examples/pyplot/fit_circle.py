"""Fit circles analytically to measure
the signed curvature of a line"""
from vedo import *

shape = Spline([
    [1.0, 2.0, -1.0],
    [1.5, 0.0,  0.4],
    [2.0, 4.0,  0.5],
    [4.0, 1.5, -0.3]],
    res=200,
)

n = 5  # nr. of points to use for the fit
npt = shape.npoints

points = shape.vertices
fitpts, circles, curvs = [], [], [0]*npt

for i in range(n, npt - n-1):
    pts = points[i-n:i+n]
    center, R, normal = fit_circle(pts)
    z = cross(pts[-1]-pts[0], center-pts[0])[2]
    curvs[i] = sqrt(1/R) * z/abs(z)
    if R < 0.75:
        circle = Circle(center, r=R).wireframe()
        circle.reorient([0,0,1], normal)
        circles.append(circle)
        fitpts.append(center)

shape.lw(8).cmap('coolwarm', curvs).add_scalarbar3d(title=':pm1/:sqrtR', c='w')
# use this trick to make the scalarbar3d become a 2d screen object:
shape.scalarbar = shape.scalarbar.clone2d("bottom-right", 0.2)

show(shape, circles, Points(fitpts, c='white'), __doc__, axes=1, bg='bb').close()
