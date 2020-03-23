"""Triangulate arbitrary line contours in 2D.
The contours may be concave, and even contain holes.
"""
from vtkplotter import *

# let's create two bidimentional contour lines
s1 = Star(line=1, n=9)
s2 = Star(line=1, n=5, r1=0.3, r2=0.4).x(0.12)

# merge the 2 lines and triangulate the inner region
sm = merge(s1, s2).triangulate().c('lightsalmon').lw(0.1)

comment = Text2D(__doc__)

show([(s1,s2,comment), sm], N=2, axes=1)
