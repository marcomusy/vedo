"""Generate 3 random sets of points
and align them using Procrustes method"""
from random import uniform as u
from vedo import Plotter, procrustesAlignment, Points


N = 15  # number of points
x = 1.0  # add some randomness

pts1 = [(u(0, x), u(0, x), u(0, x) + i) for i in range(N)]
pts2 = [(u(0, x) + 3, u(0, x) + i / 2 + 2, u(0, x) + i + 1) for i in range(N)]
pts3 = [(u(0, x) + 4, u(0, x) + i / 4 - 3, u(0, x) + i - 2) for i in range(N)]

vpts1 = Points(pts1, c="r", r=8)
vpts2 = Points(pts2, c="g", r=8)
vpts3 = Points(pts3, c="b", r=8)

# find best alignment among the n sets of Points,
# return an Assembly object formed by the aligned sets
aligned = procrustesAlignment([vpts1, vpts2, vpts3])
#print([aligned.transform])

plt = Plotter(shape=[1,2], axes=2, sharecam=False)
plt.at(0).show(vpts1, vpts2, vpts3, __doc__)
plt.at(1).show(aligned)
plt.interactive().close()
