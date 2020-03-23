"""
For each point finds the 9 closest ones and fit a sphere
color points based on the size of the sphere radius
"""
from __future__ import division, print_function
from vtkplotter import *

vp = Plotter(verbose=0, axes=0)

s = vp.load(datadir+"cow.vtk", alpha=0.3)  # .subdivide()

pts1, pts2, vals, cols = [], [], [], []

for i, p in enumerate(s.points()):
    pts = s.closestPoint(p, N=12)  # find the N closest points to p
    sph = fitSphere(pts)  # find the fitting sphere
    if sph is None:
        continue

    value = sph.info["radius"] * 10
    color = colorMap(value, "jet", 0, 1)  # map value to a RGB color
    n = versor(p - sph.info["center"])  # unit vector from sphere center to p
    vals.append(value)
    cols.append(color)
    pts1.append(p)
    pts2.append(p + n / 8)
    if not i % 500:
        print(i, "/", s.N())

vp += Points(pts1, c=cols)
vp += Lines(pts1, pts2, c="black 0.2")
vp += histogram(vals, bins=20, vrange=[0, 1]).pos(-1,1,-1)
vp += Text2D(__doc__, pos=1)

vp.show(axes=1)
