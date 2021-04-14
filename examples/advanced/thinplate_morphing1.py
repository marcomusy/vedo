"""Thin Plate Spline transformations describe a nonlinear warp
transform defined by a set of source and target landmarks.
Any point on the mesh close to a source landmark will
be moved to a place close to the corresponding target landmark.
The points in between are interpolated using Bookstein's algorithm"""
from vedo import Mesh, Points, show, dataurl
import numpy as np

np.random.seed(1)

mesh = Mesh(dataurl+"shuttle.obj").c('silver')

# pick 4 random points
indxs = np.random.randint(0, mesh.N(), 4)
pts = mesh.points()[indxs]

# and move them randomly by a little
ptsource, pttarget = [], []
for ptold in pts:
    ptnew = ptold + np.random.rand(3) * 0.2
    ptsource.append(ptold)
    pttarget.append(ptnew)
    # print(ptold,'->',ptnew)

warped = mesh.clone().thinPlateSpline(ptsource, pttarget).color("b", 0.4)

apts = Points(ptsource, r=15, c="r")

show(mesh, warped, apts, __doc__, viewup="z", axes=1)
