"""Generate two random sets of points and align
them using the Iterative Closest Point algorithm"""
from random import uniform as u
from vedo import Points, Arrows, Plotter

N1 = 25  # number of points of first set
N2 = 35  # number of points of second set
x = 1.0  # add some randomness

pts1 = [(u(0, x), u(0, x), u(0, x) + i) for i in range(N1)]
pts2 = [(u(0, x)+3, u(0, x)+i/3+2, u(0, x)+i+1) for i in range(N2)]

vpts1 = Points(pts1, r=8, c="blue5")
vpts2 = Points(pts2, r=8, c="red5")

# Find best alignment between the 2 sets of Points,
# e.i. find how to move vpts1 to best match vpts2
aligned_pts1 = vpts1.clone().alignTo(vpts2, invert=False)

# draw arrows to see where points end up
arrows = Arrows(pts1, aligned_pts1, s=0.7, c='black', alpha=0.2)

plt = Plotter(N=2, axes=1)
plt.at(0).show(vpts1, vpts2, __doc__, viewup="z")
plt.at(1).show(aligned_pts1, arrows, vpts2)
plt.interactive().close()

