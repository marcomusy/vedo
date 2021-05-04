"""linInterpolate():
[(0, 0, 0), (2, 2, 0)] # at these positions,
[(0.2,0,0), (0,0,0.2)] # these are the specified vectors
"""
from vedo import *

positions  =  [(0, 0, 0), (2, 2, 0)] # at these positions,
directions =  [(0.2,0,0), (0,0,0.2)] # these are the specified vectors

# now use linInterpolate to interpolate linearly any other point in space
# (points far from both positions will get close to the directions average)
arrs = []
for x in range(0,10):
    for y in range(0,10):
        p = [x/5, y/5, 0]
        v = linInterpolate(p, positions, directions)
        arrs.append(Arrow(p, p+v, s=0.001))

show(arrs, __doc__, axes=1).close()