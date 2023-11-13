"""Fit a surface to a set of points"""
# Thin Plate Spline transformations describe a nonlinear warp
# transform defined by a set of source and target landmarks.
# Any point on the mesh close to a source landmark will
# be moved to a place close to the corresponding target landmark.
# The points in between are interpolated using Bookstein's algorithm.
from vedo import Grid, Points, Arrows, show
import numpy as np
np.random.seed(1)

surf = Grid([0,0,0], res=[25,25])
ids = np.random.randint(0, surf.npoints, 10)  # pick 10 indices
pts = surf.vertices[ids]

ptsource, pttarget = [], []
for pt in pts:
    pt1 = pt + [0, 0, np.random.randn() * 0.1]
    pt2 = surf.closest_point(pt1)
    ptsource.append(pt2)
    pttarget.append(pt1)

warped = surf.warp(ptsource, pttarget, mode='2d')
warped.color("b4").lc('lightblue').lw(1).wireframe(False)

apts = Points(pttarget).point_size(15).c("red5")
arrs = Arrows(ptsource, pttarget).c("black")

show(warped, apts, arrs, __doc__, axes=1, viewup="z").close()
