"""
Kochanek–Bartels spline
"""
from vtkplotter import Text, Points, KSpline, show
from random import uniform as u

pts = [(u(0, 2), u(0, 2), u(0, 2) + i) for i in range(8)]

Points(pts, r=10)
Text(__doc__)

for i in range(10):
    g = (i/10-0.5)*2 # from -1 to 1
    KSpline(pts, continuity=g, tension=0, bias=0, closed=False).color(i)

# plot all object sofar created:
show(..., viewup="z", axes=1)

