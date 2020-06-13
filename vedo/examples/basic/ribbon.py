"""Form a surface by joining two lines"""
from vedo import *

l1 = [[sin(x), cos(x), x/3] for x in arange(0,9, 0.1)]
l2 = [[sin(x)+0.2, cos(x) + x/15, x/3] for x in arange(0,9, 0.1)]

t1 = Tube(l1, c="g", r=0.02)
t2 = Tube(l2, c="b", r=0.02)

r12 = Ribbon(l1, l2, res=(200,5)).alpha(0.5)
show(r12, t1, t2, __doc__, at=0, N=2, axes=1, viewup="z")

r1  = Ribbon(l1, width=0.1).alpha(0.5).color('orange')
show(r1, t1, "..or along a single line", at=1, interactive=True)

