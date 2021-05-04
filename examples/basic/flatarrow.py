"""Use 2 lines to define a flat arrow"""
from vedo import *
from numpy import arange

arrs = []
for i in range(10):
    s, c = sin(i), cos(i)
    l1 = [[sin(x)+c,     -cos(x)+s,        x] for x in arange(0,3, 0.1)]
    l2 = [[sin(x)+c+0.1, -cos(x)+s + x/15, x] for x in arange(0,3, 0.1)]
    arrs.append(FlatArrow(l1, l2, c=i, tipSize=1, tipWidth=1))

# three points, aka ellipsis, retrieves the list of all created actors
show(arrs, __doc__, viewup="z", axes=1).close()
