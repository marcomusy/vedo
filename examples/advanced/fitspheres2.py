"""For each point finds the 12 closest ones and fit a sphere.
Color points from the size of the sphere radius."""
from vedo import *
from vedo.pyplot import histogram

plt = Plotter()

msh = Mesh(dataurl+"cow.vtk").c("cyan7")

pts1, pts2, vals = [], [], []

msh_points = msh.vertices
for i in range(0, msh.npoints, 10):
    p = msh_points[i]
    pts = msh.closest_point(p, n=12)  # find the n-closest points to p
    sph = fit_sphere(pts)             # find the fitting sphere
    if sph is None:
        continue

    value = sph.radius * 10
    n = versor(p - sph.center)  # unit vector from sphere center to p
    vals.append(value)
    pts1.append(p)
    pts2.append(p + n / 8)

plt += msh, Points(pts1), Lines(pts1, pts2).c("black")
plt += histogram(vals, xtitle='radius', xlim=[0,2]).clone2d("bottom-left")
plt += __doc__
plt.show().close()
