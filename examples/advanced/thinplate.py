'''
Thin Plate Spline transformations describe a nonlinear warp 
transform defined by a set of source and target landmarks. 
Any point on the mesh close to a source landmark will 
be moved to a place close to the corresponding target landmark. 
The points in between are interpolated using Bookstein's algorithm.   
'''
from vtkplotter import Plotter, thinPlateSpline, points, text
import numpy as np
np.random.seed(1)

vp = Plotter(verbose=0)

act = vp.load('data/shuttle.obj')

# pick 4 random points 
indxs = np.random.randint(0, act.N(), 4)

# and move them randomly by a little
ptsource, pttarget = [], []
for i in indxs:
    ptold = act.point(i)
    ptnew = ptold + np.random.rand(3)*0.2
    act.point(i, ptnew)
    ptsource.append(ptold)
    pttarget.append(ptnew)
    print(ptold,'->',ptnew)

warped = thinPlateSpline(act, ptsource, pttarget)
warped.alpha(0.4).color('b')
#print(warped.info['transform']) # saved here.

apts = points(ptsource, r=15, c='r')

vp.show([act, warped, apts, text(__doc__)], viewup='z')

