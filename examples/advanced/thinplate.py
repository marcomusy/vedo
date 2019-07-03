"""
Thin Plate Spline transformations describe a nonlinear warp
transform defined by a set of source and target landmarks.
Any point on the mesh close to a source landmark will
be moved to a place close to the corresponding target landmark.
The points in between are interpolated using Bookstein's algorithm.
"""
from vtkplotter import *
import numpy as np

np.random.seed(1)

vp = Plotter(axes=1)

act = vp.load(datadir+"shuttle.obj", c='silver')

# pick 4 random points
indxs = np.random.randint(0, act.N(), 4)

# and move them randomly by a little
ptsource, pttarget = [], []
for i in indxs:
    ptold = act.getPoint(i)
    ptnew = ptold + np.random.rand(3) * 0.2
    act.setPoint(i, ptnew)
    ptsource.append(ptold)
    pttarget.append(ptnew)
    # print(ptold,'->',ptnew)

warped = thinPlateSpline(act, ptsource, pttarget)
warped.alpha(0.4).color("b")
# print(warped.info['transform']) # saved here.

apts = Points(ptsource, r=15, c="r")

vp.show(act, warped, apts, Text(__doc__), viewup="z")
