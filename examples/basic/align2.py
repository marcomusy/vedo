"""Generate two random sets of points and align
them using the Iterative Closest Point algorithm"""
from random import uniform as u
from vedo import settings, Points, Arrows, Plotter

settings.default_font = "Calco"

N1 = 25  # number of points of first set
N2 = 35  # number of points of second set
x = 1.0  # add some randomness

# Create two sets of random points with different colors
pts1 = [(u(0, x), u(0, x), u(0, x) + i) for i in range(N1)]
pts2 = [(u(0, x)+3, u(0, x)+i/3+2, u(0, x)+i+1) for i in range(N2)]
vpts1 = Points(pts1).ps(10).c("blue5")
vpts2 = Points(pts2).ps(10).c("red5")

# Find best alignment between the 2 sets of Points,
# e.i. find how to move vpts1 to best match vpts2
aligned_pts1 = vpts1.clone().align_to(vpts2, invert=False)
txt = aligned_pts1.transform.__str__()

# Create arrows to visualize how the points move during alignment
arrows = Arrows(pts1, aligned_pts1, s=0.7).c("black")

# Create a plotter with two subplots
plt = Plotter(N=2, axes=1)
plt.at(0).show(__doc__, vpts1, vpts2)
plt.at(1).show(txt, aligned_pts1, arrows, vpts2, viewup="z")
plt.interactive()
plt.close()

