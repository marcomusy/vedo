# In this example we modify the mesh of a shape
# by moving the points along the normals to the surface
# and along the radius of a sphere centered at the center of mass.
# At each step we redefine the actor so that the normals are
# recalculated for the underlying polydata.
from __future__ import division, print_function
from plotter import vtkPlotter, vector, norm, mag

vp = vtkPlotter(axes=0)

s = vp.load('data/290.vtk', wire=1)
vp.actors.append( s.clone(c='red 1.0', wire=0) )

c = vp.centerOfMass(s)
vp.point(c)

Niter = 4
for t in range(Niter):
    print('iteration', t)
    coords = s.coordinates()
    normals= s.normals()
    aves = vp.averageSize(s)*1.5

    for i in range(s.N()):
        n = normals[i]
        p = coords[i]
        q = norm(p-c)*aves +c
        dp = mag(q-p)
        alongn = n*dp
        alongr = q - p
        newp = p + (alongn + alongr) /2 /Niter 
        s.point(i, newp)
        
    #refresh actor, so to recalc polydata normals
    s = vp.makeActor(s.polydata(), wire=1, alpha=.1)
    vp.actors.append(s)

vp.show()