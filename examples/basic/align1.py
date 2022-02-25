"""Align 2 shapes:
the red line to the yellow surface"""
from vedo import *

limb = Mesh(dataurl + "270.vtk").c("gold").flag()
rim  = Mesh(dataurl + "270_rim.vtk").c("red").lw(4)

# Make a clone copy of rim and align it to limb
# rigid=True doesn't allow scaling
rim2 = rim.clone().alignTo(limb, rigid=True)
rim2.c("green").lw(5)

d = 0
for p in rim2.points():
    cpt = limb.closestPoint(p)
    d += mag2(p - cpt)  # square of residual distance

printc("ave. squared distance =", d/rim2.N())
# vtkTransform is available through:
#printc([rim2.transform])

show(limb, rim, rim2, __doc__, axes=1).close()
