"""
In this example we modify the mesh of a shape
by moving the points along the normals to the surface
and along the radius of a sphere centered at the center of mass.
At each step we redefine the actor so that the normals are
recalculated for the underlying polydata.
"""
from __future__ import division, print_function
from vtkplotter import Plotter, versor, mag, settings, Point, Text

settings.computeNormals = True  # on object creation by default

vp = Plotter(axes=0, verbose=0, bg="w")

s = vp.load("data/290.vtk", c="red", bc="plum")
c = s.centerOfMass()
vp.add(Point(c))

Niter = 4
for t in range(Niter):
    print("iteration", t)
    coords = s.coordinates()
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
    s.alpha(0.1).color("gold").wire(True)
    vp.add(s)

vp.add(Text(__doc__, c="k"))
vp.show()
