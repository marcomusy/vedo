"""
Align 2 shapes:
the red line to the yellow surface
"""
from vtkplotter import *

# flag() shows the filename when hovering with mouse
limb = load(datadir + "270.vtk").c("gold").flag()
rim  = load(datadir + "270_rim.vtk").c("red").lw(4)

# rigid=True doesn't allow scaling
arim = alignICP(rim, limb, rigid=True).c("green").lw(5)

d = 0
for p in arim.points():
    cpt = limb.closestPoint(p)
    d += mag2(p - cpt)  # square of residual distance

printc("ave. squared distance =", d / arim.N(), c="g")
printc("vtkTransform is available with getTransform():")
printc([arim.getTransform()])
show(limb, rim, arim, Text2D(__doc__))
