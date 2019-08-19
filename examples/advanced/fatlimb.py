"""
In this example we modify the mesh of a shape
by moving the points along the normals to the surface
and along the radius of a sphere centered at the center of mass.
At each step we redefine the actor so that the normals are
recalculated for the underlying polydata.
"""
from __future__ import division, print_function
from vtkplotter import *

settings.computeNormals = True  # on object creation by default

vp = Plotter(axes=0, verbose=0, bg="w")

s = vp.load(datadir+"290.vtk", c="red")
c = s.centerOfMass()
vp += [Point(c), Text(__doc__, c="k")]

Niter = 4
for t in range(Niter):
    print("iteration", t)
    coords = s.getPoints()
    normals = s.normals()
    aves = s.averageSize() * 1.5

    for i in range(s.N()):
        n = normals[i]
        p = coords[i]
        q = versor(p - c) * aves + c  # versor = unit vector
        dp = mag(q - p)
        alongn = n * dp
        alongr = q - p  # bias normal
        newp = p + (alongn + alongr) / 2 / Niter
        s.setPoint(i, newp)

    # refresh actor, so polydata normals are recalculated
    s = s.clone()
    vp += s.alpha(0.1).color("gold").wireframe(True) #add into Plotter

vp.show()
