from __future__ import division, print_function
from plotter import vtkPlotter, vector, norm, mag

vp = vtkPlotter(axes=0)

s = vp.load('data/290.vtk',wire=1)
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
        alongn = n*dp/Niter
        alongr = (q-p)/Niter
        newp =  p + (alongn + alongr) /2
        s.point(i, newp)
        
    #refresh actor polydata, so to recalc normals
    s = vp.makeActor(s.polydata(), wire=1, alpha=.1)
    vp.actors.append(s)

vp.show()