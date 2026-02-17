"""Align three random point sets with Procrustes analysis."""
from random import uniform as u
from vedo import Plotter, procrustes_alignment, Points

N = 15
x = 1.0

# Generate three noisy point sets with different offsets.
pts1 = [(u(0, x), u(0, x), u(0, x) + i) for i in range(N)]
pts2 = [(u(0, x) + 3, u(0, x) + i / 2 + 2, u(0, x) + i + 1) for i in range(N)]
pts3 = [(u(0, x) + 4, u(0, x) + i / 4 - 3, u(0, x) + i - 2) for i in range(N)]

# Convert to renderable point clouds.
vpts1 = Points(pts1).c("r").ps(8)
vpts2 = Points(pts2).c("g").ps(8)
vpts3 = Points(pts3).c("b").ps(8)

# Procrustes returns an Assembly containing aligned copies.
aligned = procrustes_alignment([vpts1, vpts2, vpts3])

# Compare original sets (left) and aligned sets (right).
plt = Plotter(shape=[1, 2], axes=2, sharecam=False)
plt.at(0).show(vpts1, vpts2, vpts3, __doc__)
plt.at(1).show(aligned)
plt.interactive().close()
