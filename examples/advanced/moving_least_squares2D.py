#  This example shows how to use a variant of the 
# Moving Least Squares (MLS) algorithm to project a cloud 
# of 20k points to become a smooth surface.
# The parameter f controls the size of the local regression.
# The input actor's polydata is modified by the method
# so more than one pass is possible.
# If showNPlanes>0 an actor is built demonstrating the 
# details of the regression for some random points
#  In the second window we show the error estimated for
# each point in color scale (left) or in size scale (right).
#
from __future__ import division, print_function
from plotter import vtkPlotter, cos, sqrt, colorMap, vector
from random import gauss
import numpy as np


vp1 = vtkPlotter(shape=(1,4), axes=0)

# Generate a random cloud of 20k points in space
def fnc(p): 
	x,y,z = p
	return vector(x,y, z+cos(x*y)/1.5) #whatever function..
pts = []
for i in range(20000):
	x,y,z = gauss(0,1), gauss(0,1), gauss(0,1)
	r = sqrt(x**2+y**2+z**2)/3
	p = [x/r, y/r, z/r]  #generates a point on a sphere
	pts.append( fnc(p)*gauss(1, 0.07) ) #modify and randomize point

# this is the true generated object without noise:
act = vp1.sphere(r=3, c='k 0.05', wire=1, res=60, legend='true object') 
for i,p in enumerate(act.coordinates()): act.point(i, fnc(p))


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
vp1.show(s1,   at=3, legend='third pass') 


#################################### draw errors
vp2 = vtkPlotter(shape=(1,2), axes=3)

vmin,vmax = np.min(s1.variances), np.max(s1.variances)
print('min and max of variances:', vmin,vmax)
vcols = [ colorMap(v/vmax) for v in s1.variances ] # scalars->colors

a0= vp2.spheres(s1.coordinates(), c=vcols, r=0.07, legend='variance')
a1= vp2.spheres(s1.coordinates(), c='red', r=s1.variances, legend='variance')

vp2.show(a0, at=0)
vp2.show([a1,act], at=1, interactive=1)
