"""
This example shows how to use a variant of the
Moving Least Squares (MLS) algorithm to project a cloud
of points to become a smooth surface.
The parameter f controls the size of the local regression.
The input actor's polydata is modified by the method
so more than one pass is possible.
If showNPlanes>0 an actor is built demonstrating the
details of the regression for some random points
In the second window we show the error estimated for
each point in color scale (left) or in size scale (right).
"""
from __future__ import division, print_function
from vtkplotter import *
import numpy as np

vp1 = Plotter(shape=(1, 4), axes=4, bg="w")

act = vp1.load(datadir+"bunny.obj").normalize().subdivide()
act.color("k").alpha(0.05).wire(True)
pts = act.coordinates(copy=True)  # pts is a copy of the points not a reference
pts += np.random.randn(len(pts), 3) / 40  # add noise, will not mess up the original points


#################################### smooth cloud with MLS
# build the points actor
s0 = Points(pts, c="blue", r=3).legend("point cloud")
vp1.show(s0, at=0)

s1 = s0.clone().color("dg")  # a dark green copy of s0

# project s1 points into a smooth surface of points
# return a demo actor showing 30 regressions at random points
mls1 = smoothMLS2D(s1, f=0.5, showNPlanes=30) #first pass
vp1.show(mls1, at=1)

mls2 = smoothMLS2D(s1, f=0.3, showNPlanes=30) # second pass
vp1.show(mls2, at=2)

mls3 = smoothMLS2D(s1, f=0.1).legend("third pass")
vp1.show(s1, at=3)


#################################### draw errors
vp2 = Plotter(pos=(200, 400), shape=(1, 2), axes=4, bg="w")

variances = s1.info["variances"]
vmin, vmax = np.min(variances), np.max(variances)
print("min and max of variances:", vmin, vmax)
vcols = [colorMap(v, "jet", vmin, vmax) for v in variances]  # scalars->colors

a0 = Spheres(s1.coordinates(), c=vcols, r=0.03).legend("variance")
a1 = Spheres(s1.coordinates(), c="red", r=variances).legend("variance")

vp2.show(a0, Text(__doc__, c="k"), at=0)
vp2.show(a1, act, at=1, zoom=1.3, interactive=1)
