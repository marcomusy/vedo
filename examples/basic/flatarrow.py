"""Use 2 lines to define a flat arrow"""
from vedo import *
from numpy import arange

arrs = []
for i in range(10):
    s, c = sin(i), cos(i)
    l1 = [[sin(x)+c,     -cos(x)+s,        x] for x in arange(0,3, 0.1)]
    l2 = [[sin(x)+c+0.1, -cos(x)+s + x/15, x] for x in arange(0,3, 0.1)]
    farr = FlatArrow(l1, l2, tip_size=1, tip_width=1).c(i)
    arrs.append(farr)

show(arrs, __doc__, viewup="z", axes=1).close()
