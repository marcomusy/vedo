"""Generate two random sets of points and align
them using the Iterative Closest Point algorithm"""
from random import uniform as u
from vedo import *

N1 = 25  # number of points of first set
N2 = 35  # number of points of second set
x = 1.0  # add some randomness

pts1 = [(u(0, x), u(0, x), u(0, x) + i) for i in range(N1)]
pts2 = [(u(0, x)+3, u(0, x)+i/3+2, u(0, x)+i+1) for i in range(N2)]

vpts1 = Points(pts1, r=8, c="b")
vpts2 = Points(pts2, r=8, c="r")

# Find best alignment between the 2 sets of Points,
# e.i. find how to move vpts1 to best match vpts2
aligned_pts1 = vpts1.clone().alignTo(vpts2, invert=False)

# draw arrows to see where points end up
arrows = Arrows(pts1, aligned_pts1, s=0.7, alpha=0.2).c("k")

show(vpts1, vpts2, __doc__, at=0, N=2, axes=1)

show(aligned_pts1, arrows, vpts2, at=1, interactive=True).close()





