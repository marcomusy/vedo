"""
A scalar array can be specified to vary 
radius and color of a line represented as a tube.
"""
print(__doc__)

from vtkplotter import *

doc = Text(__doc__)


ln = [[sin(x), cos(x), x / 2] for x in arange(0, 9, 0.1)]
N = len(ln)

############################### a simple tube( along ln
t1 = Tube(ln, c="blue", r=0.08)
show(t1, doc, at=0, N=3, axes=1, viewup="z")

############################### vary radius
rads = [0.3 * (cos(6.0 * ir / N)) ** 2 + 0.1 for ir in range(N)]

t2 = Tube(ln, r=rads, c="tomato", res=24)
show(t2, at=1)

############################### vary color
cols = [i for i in range(N)]
cols = makeBands(cols, 5)  # make color bins

t3 = Tube(ln, r=rads, c=cols, res=24)
show(t3, at=2, interactive=1)
