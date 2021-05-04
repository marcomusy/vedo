"""Generate 3 random sets of points and align
them using Procrustes method."""
from __future__ import division, print_function
from random import uniform as u

from vedo import Plotter, procrustesAlignment, Points

vp = Plotter(shape=[1, 2], axes=2, sharecam=0)

N = 15  # number of points
x = 1.0  # add some randomness

pts1 = [(u(0, x), u(0, x), u(0, x) + i) for i in range(N)]
pts2 = [(u(0, x) + 3, u(0, x) + i / 2 + 2, u(0, x) + i + 1) for i in range(N)]
pts3 = [(u(0, x) + 4, u(0, x) + i / 4 - 3, u(0, x) + i - 2) for i in range(N)]

vpts1 = Points(pts1, c="r").legend("set1")
vpts2 = Points(pts2, c="g").legend("set2")
vpts3 = Points(pts3, c="b").legend("set3")

vp.show(vpts1, vpts2, vpts3, __doc__, at=0)

# find best alignment among the n sets of Points,
# return an Assembly object formed by the aligned sets
aligned = procrustesAlignment([vpts1, vpts2, vpts3])

# print(aligned.info['transform'])

vp.show(aligned, at=1, interactive=1).close()
