'''
Align 2 shapes and for each vertex of the first draw
and arrow to the closest point of the second.
The default method is the Iterative Closest Point algorithm.
The source transformation is saved in actor.info['transform']
rigid=True doesn't allow scaling
'''
from vtkplotter import Plotter, printc, mag2, align, Arrow, Text

vp = Plotter(verbose=0, axes=4)

vp.add(Text(__doc__)) # add comment above

limb = vp.load('data/270.vtk')
rim  = vp.load('data/270_rim.vtk')
rim.color('r').lineWidth(4)

arim = align(rim, limb, iters=100, rigid=True)
vp.add(arim.color('g').lineWidth(4))

d = 0
prim = arim.coordinates()
for p in prim: 
    cpt = limb.closestPoint(p)
    vp.add(Arrow(p, cpt, c='g'))
    d += mag2(p-cpt) # square of residual distance

printc("ave. squared distance =", d/len(prim), c='g')
printc("vtkTransform is stored in actor.info['transform']:")
printc([arim.info['transform']])
vp.show()
