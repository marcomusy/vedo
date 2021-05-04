"""Triangulate arbitrary line contours in 2D.
The contours may be concave, and even contain holes.
"""
from vedo import *

# let's create two bidimentional contour lines
s1 = Star(line=True, n=9)
s2 = Star(line=True, n=5, r1=0.3, r2=0.4).x(0.12)
# merge the 2 lines and triangulate the inner region
sm = merge(s1, s2).triangulate().c('lightsalmon').lw(0.1)

show([(s1,s2,__doc__), sm], N=2, axes=1).close()
