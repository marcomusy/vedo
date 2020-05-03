"""Fit a plane to regions of a surface defined by
N points that are closest to a given point of the surface.
For some of these point we show the fitting plane.
Black points are the N points used for fitting.
Green histogram is the distribution of residuals from the fitting.
"""
from vtkplotter import *
from vtkplotter.pyplot import histogram

vp = Plotter()
vp += __doc__

s = load(datadir+"cow.vtk").subdivide().normalize().alpha(0.3)
vp += s

variances = []
for i, p in enumerate(s.points()):
    if i % 100:
        continue  # skip most points
    pts = s.closestPoint(p, N=12)  # find the N closest points to p
    plane = fitPlane(pts)          # find the fitting plane
    vp += plane
    vp += Points(pts)              # blue points
    vp += Arrow(plane.center, plane.center+plane.normal/10, c="g")
    variances.append(plane.variance)

vp += histogram(variances).scale(25).pos(.6,-.3,-.9)

vp.show(viewup="z")
