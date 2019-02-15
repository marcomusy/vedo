'''
Thin Plate Spline transformations describe a nonlinear warp
transform defined by a set of source and target landmarks.
Any point on the mesh close to a source landmark will
be moved to a place close to the corresponding target landmark.
The points in between are interpolated using Bookstein's algorithm.
'''

from vtkplotter import *
import numpy as np
np.random.seed(2)

grids = []
for i in range(10):
    grids.append(Grid([0, 0, i/10.], resx=100, resy=100))
act = mergeActors(grids)  # merge grids into a single object

idxs = np.random.randint(0, act.N(), 10)  # pick 10 indexes

ptsource, pttarget = [], []
for i in idxs:
    ptold = act.getPoint(i) + np.random.randn(3)*0.02
    ptsource.append(ptold)
    ptnew = ptold + [0, 0, np.random.randn(1)*0.10]  # move in z
    pttarget.append(ptnew)

warped = thinPlateSpline(act, ptsource, pttarget)
warped.alpha(0.2).color('b')
# print(warped.info['transform']) # saved here.

apts = Points(ptsource, r=5, c='r')
arrs = Arrows(ptsource, pttarget)

show([warped, apts, arrs, Text(__doc__)],
     axes=9, viewup='z', verbose=0, bg='w')

