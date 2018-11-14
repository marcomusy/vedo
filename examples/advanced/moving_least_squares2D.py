#  This example shows how to use a variant of the 
# Moving Least Squares (MLS) algorithm to project a cloud 
# of points to become a smooth surface.
# The parameter f controls the size of the local regression.
# The input actor's polydata is modified by the method
# so more than one pass is possible.
# If showNPlanes>0 an actor is built demonstrating the 
# details of the regression for some random points
#  In the second window we show the error estimated for
# each point in color scale (left) or in size scale (right).
#
from __future__ import division, print_function
from vtkplotter import Plotter, colorMap
from vtkplotter.analysis import smoothMLS2D
import numpy as np


vp1 = Plotter(shape=(1,4), axes=4)

act = vp1.load('data/shapes/bunny.obj', c='k 0.05', wire=1).normalize().subdivide()
pts = act.coordinates(copy=True)      # pts is a copy of the points not a reference
pts += np.random.randn(len(pts),3)/40 # add noise, will not mess up the original points


#################################### smooth cloud with MLS
# build the points actor
s0 = vp1.points(pts, c='blue', r=3, legend='point cloud') 
vp1.show(s0, at=0) 

#print(vp1.renderWin.GetScreenSize (), vp1.renderWin.SetPosition (60, 60))

s1 = s0.clone(c='dg')                 # a dark green copy of s0

# project s1 points into a smooth surface of points 
# return a demo actor showing 30 regressions at random points
mls1 = smoothMLS2D(s1, f=0.5, showNPlanes=30) # first pass
vp1.show(mls1, at=1, legend='first pass') 

mls2 = smoothMLS2D(s1, f=0.3, showNPlanes=30) # second pass
vp1.show(mls2, at=2, legend='second pass') 

mls3 = smoothMLS2D(s1, f=0.1)                 # third pass
vp1.show(s1,   at=3, legend='third pass', zoom=1.3) 


#################################### draw errors
vp2 = Plotter(pos=(200,400), shape=(1,2), axes=4)

vmin,vmax = np.min(s1.variances), np.max(s1.variances)
print('min and max of variances:', vmin,vmax)
vcols = [ colorMap(v, 'jet', vmin, vmax) for v in s1.variances ] # scalars->colors

a0= vp2.spheres(s1.coordinates(), c=vcols, r=0.03, legend='variance')
a1= vp2.spheres(s1.coordinates(), c='red', r=s1.variances, legend='variance')

vp2.show(a0, at=0)
vp2.show([a1, act], at=1, zoom=1.3, interactive=1)
