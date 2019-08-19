"""
Align 2 shapes and for each vertex of the first draw
and arrow to the closest point of the second.
The source transformation is accessible with getTransform()
rigid=True doesn't allow scaling
"""
from vtkplotter import *

vp = Plotter()

limb = vp.load(datadir + "270.vtk")
rim  = vp.load(datadir + "270_rim.vtk").c("r").lw(4)

arim = alignICP(rim, limb, rigid=True).c("g").lw(5)
vp += [arim, Text(__doc__)]

d = 0
for p in arim.getPoints():
    cpt = limb.closestPoint(p)
    vp += Arrow(p, cpt, c="g")
    d += mag2(p - cpt)  # square of residual distance

printc("ave. squared distance =", d / arim.N(), c="g")
printc("vtkTransform is available with getTransform():")
printc([arim.getTransform()])
vp.show()
