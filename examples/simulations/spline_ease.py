"""Spline three points in space"""
from vedo import *
import numpy as np

pts = [[0,0,0],
       [0.5,0.6,0.8],
       [1,1,1]]
gpts = Points(pts, r=10).c('green',0.5)

# Create a spline where the final points are more dense (easing)
line = Spline(pts, easing="OutCubic", res=100)

vpts = line.clone().pointSize(3).shift(0,0.1,0) # a dotted copy

# Calculate positions as a fraction of the length of the line,
# being x=0 the first point and x=1 the last point.
# This corresponds to an imaginary point that travels along the line
# at constant speed:
equi_pts = Points([line.eval(x) for x in np.arange(0,1, 0.1)]).c('blue')

redpt = Point(r=25).c('red')
plt = show(vpts, gpts, line, redpt, equi_pts, axes=1, interactive=0)
# Animation
for i in range(line.N()):
    x = line.points(i)
    redpt.pos(x) # assign the new position
    plt.render()
plt.interactive().close()
