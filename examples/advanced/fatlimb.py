"""Modify the mesh of a shape by moving the points
along the normals to the surface and along the
radius of a sphere centered at the center of mass"""
from vedo import *

plt = Plotter()
s = plt.load(dataurl+"290.vtk").subdivide()
s.c("red").bc("lightblue")

cn = s.centerOfMass()
plt += [Point(cn), __doc__]

Niter = 4
for t in range(Niter):
    print("iteration", t)
    s = s.clone()

    coords = s.points()
    normals = s.normals()
    aves = s.averageSize() * 1.5

    newpts = []
    for i in range(s.N()):
        n = normals[i]
        p = coords[i]
        q = versor(p - cn) * aves + cn  # versor = vector of norm 1
        dp = mag(q - p)
        alongn = n * dp
        alongr = q - p  # bias normal
        newp = p + (alongn + alongr) / 4.0 / Niter
        newpts.append(newp)

    s.points(newpts).computeNormals() # set the new points of the mesh
    s.alpha(0.1).color("gold").wireframe(True) # cosmetics

    plt += s # add into Plotter

plt.show(axes=11).close()
