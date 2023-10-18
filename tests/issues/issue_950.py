from vedo import *

np.random.seed(1)

# generate a random set of points in 3D space
pts1 = np.random.randn(100, 3)
pts2 = np.random.randn(100, 3)

lines = []
for i in range(100):
    # generate a line between two points
    lines.append(Line(pts1[i], pts2[i]).color(i))

# create a new line from the lines list
newline = Lines(lines)

show([lines, newline], N=2, axes=1)