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
from vtkplotter import Plotter, cos, sqrt, colorMap, vector
from random import gauss
import numpy as np


vp1 = Plotter(shape=(1,4), axes=0)

act = vp1.load('data/shapes/bunny.obj', c='k 0.05', wire=1).normalize().subdivide()
pts = act.coordinates()
pts += np.random.randn(len(pts),3)/40 # add noise

#################################### smooth cloud with MLS
# build the points actor
s0 = vp1.points(pts, c='blue', r=2, legend='point cloud') 
s1 = s0.clone(c='dg') # a dark green copy of it

# project s1 points into a smooth surface of points 
# return a demo actor showing 30 regressions at random points
mls1 = vp1.smoothMLS2D(s1, f=0.5, showNPlanes=30) # first pass
mls2 = vp1.smoothMLS2D(s1, f=0.3, showNPlanes=30) # second pass
vp1.smoothMLS2D(s1, f=0.1) # third pass

vp1.show([s0,act], at=0) 
vp1.show(mls1, at=1, legend='first pass') 
vp1.show(mls2, at=2, legend='second pass') 
vp1.show(s1,   at=3, legend='third pass', zoom=1.3) 

#################################### draw errors
vp2 = Plotter(shape=(1,2), axes=3)

vmin,vmax = np.min(s1.variances), np.max(s1.variances)
print('min and max of variances:', vmin,vmax)
vcols = [ colorMap(v, 'jet', vmin, vmax) for v in s1.variances ] # scalars->colors

a0= vp2.spheres(s1.coordinates(), c=vcols, r=0.03, legend='variance')
a1= vp2.spheres(s1.coordinates(), c='red', r=s1.variances, legend='variance')

vp2.show(a0, at=0)
vp2.show([a1,act], at=1, zoom=1.3, interactive=1)
