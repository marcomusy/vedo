"""For each point finds the 12 closest ones and fit a sphere.
Color points from the size of the sphere radius."""
from vedo import *
from vedo.pyplot import histogram

plt = Plotter()

s = plt.load(dataurl+"cow.vtk").alpha(0.3)

pts1, pts2, vals, cols = [], [], [], []

for i in range(0, s.N(), 10):
    p = s.points(i)
    pts = s.closestPoint(p, N=12)  # find the N closest points to p
    sph = fitSphere(pts)           # find the fitting sphere
    if sph is None:
        continue

    value = sph.radius * 10
    color = colorMap(value, "jet", 0, 1)  # map value to a RGB color
    n = versor(p - sph.center)  # unit vector from sphere center to p
    vals.append(value)
    cols.append(color)
    pts1.append(p)
    pts2.append(p + n / 8)

plt += Points(pts1, c=cols)
plt += Lines(pts1, pts2, c="black")
plt += histogram(vals, xtitle='radius', xlim=[0,2]).pos(-1,0.5,-1)
plt += __doc__

plt.show().close()
