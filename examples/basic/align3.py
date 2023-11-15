"""Generate 3 random sets of points
and align them using Procrustes method"""
from random import uniform as u
from vedo import Plotter, procrustes_alignment, Points

# Define number of points and a randomness factor
N = 15  # number of points
x = 1.0  # add some randomness

# Generate 3 sets of random points
pts1 = [(u(0, x), u(0, x), u(0, x) + i) for i in range(N)]
pts2 = [(u(0, x) + 3, u(0, x) + i / 2 + 2, u(0, x) + i + 1) for i in range(N)]
pts3 = [(u(0, x) + 4, u(0, x) + i / 4 - 3, u(0, x) + i - 2) for i in range(N)]

# Convert the sets of points into Points objects with different colors and sizes
vpts1 = Points(pts1).c("r").ps(8)
vpts2 = Points(pts2).c("g").ps(8)
vpts3 = Points(pts3).c("b").ps(8)

# Perform Procrustes alignment on the sets of points
#  and obtain the aligned sets
# return an Assembly object formed by the aligned sets
aligned = procrustes_alignment([vpts1, vpts2, vpts3])
#print([aligned.transform])

# Create a Plotter object with a 1x2 grid, 2D axes, 
#  and independent camera control
plt = Plotter(shape=[1,2], axes=2, sharecam=False)
plt.at(0).show(vpts1, vpts2, vpts3, __doc__)
plt.at(1).show(aligned)
plt.interactive().close()
