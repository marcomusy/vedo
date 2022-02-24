"""Thin Plate Spline transformations describe a nonlinear warp
transform defined by a set of source and target landmarks.
Any point on the mesh close to a source landmark will
be moved to a place close to the corresponding target landmark.
The points in between are interpolated using Bookstein's algorithm."""
from vedo import Grid, merge, Points, Arrows, show
import numpy as np
#np.random.seed(2)

grids = []
for i in range(5):
    gr = Grid([0,0,i/8], res=[12,12])
    grids.append(gr)
mesh = merge(grids)  # merge grids into a single object

idxs = np.random.randint(0, mesh.N(), 10)  # pick 10 indices
pts = mesh.points()[idxs]

ptsource, pttarget = [], []
for pt in pts:
    ptold = pt + np.random.randn(3) * 0.02
    ptsource.append(ptold)
    ptnew = ptold + [0, 0, np.random.randn(1) * 0.1]  # move in z
    pttarget.append(ptnew)

warped = mesh.warp(ptsource, pttarget)
warped.color("b4").lc('light blue').wireframe(False).lw(1)

apts = Points(ptsource, r=10, c="r")
arrs = Arrows(ptsource, pttarget, c='k')

show(warped, apts, arrs, __doc__, axes=1, viewup="z").close()
