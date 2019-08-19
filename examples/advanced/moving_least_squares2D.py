"""This example shows how to use a variant of the
Moving Least Squares (MLS) algorithm to project a cloud
of points to become a smooth surface.
In the second window we show the error estimated for
each point in color scale (left) or in size scale (right).
"""
from vtkplotter import *
import numpy as np

vp1 = Plotter(N=3, bg="w")

act = vp1.load(datadir+"bunny.obj").normalize().subdivide()

pts = act.getPoints(copy=True)  # pts is a copy of the points not a reference
pts += np.random.randn(len(pts), 3)/20  # add noise, will not mess up the original points


#################################### smooth cloud with MLS
# build the points actor
s0 = Points(pts, r=3).color("blue").legend("original\npoint cloud")
vp1.show(s0, at=0)

# project s1 points into a smooth surface of points
# return a demo actor showing 30 regressions at random points
# The parameter f controls the size of the local regression.
mls1 = smoothMLS2D(  s0, f=0.5, showNPlanes=30)  #first pass
vp1.show(mls1, at=1)

mls2 = smoothMLS2D(mls1, radius=0.1).legend("second pass")
vp1.show(mls2, at=2)


#################################### draw errors
vp2 = Plotter(pos=(300, 400), N=2, bg="w")

variances = mls2.info["variances"]
vmin, vmax = np.min(variances), np.max(variances)
print("min and max of variances:", vmin, vmax)
vcols = [colorMap(v, "jet", vmin, vmax) for v in variances]  # scalars->colors

a0 = Spheres(mls2.getPoints(), c=vcols, r=0.02) # error as color
a1 = Spheres(mls2.getPoints(), c="red", r=variances/4) # error as point size

txt = Text(__doc__, c="k")
act.color("k").alpha(0.05).wireframe()

vp2.show(a0, txt, at=0)
vp2.show(a1, act, at=1, zoom=1.3, interactive=1)
