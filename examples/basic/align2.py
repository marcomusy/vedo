"""Align two random point clouds with the ICP algorithm."""
from random import uniform as u
from vedo import settings, Points, Arrows, Plotter

settings.default_font = "Calco"

N1 = 25
N2 = 35
x = 1.0

# Build two random point sets with different offsets.
pts1 = [(u(0, x), u(0, x), u(0, x) + i) for i in range(N1)]
pts2 = [(u(0, x) + 3, u(0, x) + i / 3 + 2, u(0, x) + i + 1) for i in range(N2)]
vpts1 = Points(pts1).ps(10).c("blue5")
vpts2 = Points(pts2).ps(10).c("red5")

# Estimate the transform that maps vpts1 as close as possible to vpts2.
aligned_pts1 = vpts1.clone().align_to(vpts2, invert=False)
txt = aligned_pts1.transform.__str__()

# Arrows show how points moved after alignment.
arrows = Arrows(pts1, aligned_pts1, s=0.7).c("black")

plt = Plotter(N=2, axes=1)
plt.at(0).show(__doc__, vpts1, vpts2)
plt.at(1).show(txt, aligned_pts1, arrows, vpts2, viewup="z")
plt.interactive()
plt.close()
