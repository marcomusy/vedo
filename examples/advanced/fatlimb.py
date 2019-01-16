# In this example we modify the mesh of a shape
# by moving the points along the normals to the surface
# and along the radius of a sphere centered at the center of mass.
# At each step we redefine the actor so that the normals are
# recalculated for the underlying polydata.
#
from __future__ import division, print_function
from vtkplotter import Plotter, norm, mag

vp = Plotter(axes=0)

s = vp.load('data/290.vtk', c='red')
vp.actors.append( s.clone().color('gold').wire(True) )

c = s.centerOfMass()
vp.point(c)

Niter = 4
for t in range(Niter):
    print('iteration', t)
    coords = s.coordinates()
    normals= s.normals()
    aves = s.averageSize()*1.5

    for i in range(s.N()):
        n = normals[i]
        p = coords[i]
        q = norm(p-c)*aves + c
        dp = mag(q-p)
        alongn = n*dp
        alongr = q - p # bias normal
        newp = p + (alongn + alongr) /2 /Niter 
        s.point(i, newp)
        
    #refresh actor, so polydata normals are recalculated
    s = s.clone().alpha(0.1).color('gold').wire(True)
    vp.actors.append(s)

vp.show()