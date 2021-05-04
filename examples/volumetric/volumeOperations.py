"""
Perform other simple mathematical operation between 3d Volumes.
Possible operations are: +, -, /, 1/x, sin, cos, exp, log, abs, **2, sqrt,
  min, max, atan, atan2, median, mag, dot, gradient, divergence, laplacian.
Alphas defines the opacity transfer function in the scalar range.
"""
print(__doc__)

from vedo import *

vp = Plotter(N=6)

v0 = Volume(dataurl+'embryo.slc').c(0)
v0.addScalarBar3D()
vp.show(v0, "original", at=0)

v1 = v0.clone().operation("gradient").alpha([0,0,1,0,0,0,0])#.printHistogram(logscale=1)
v1.addScalarBar3D()
vp.show(v1, "gradient", at=1)

v2 = v0.clone().operation("divergence").c(2)
v2.addScalarBar3D()
vp.show(v2, "divergence", at=2)

v3 = v0.clone().operation("laplacian")#.c(3).alpha([0, 1, 0, 0, 1])
v3.addScalarBar3D()
vp.show(v3, "laplacian", at=3)

v4 = v0.clone().operation("median").c(4)
v4.addScalarBar3D()
vp.show(v4, "median", at=4)

v5 = v0.clone().operation("dot", v0).c(7)
v5.addScalarBar3D()
vp.show(v5, "dot(v0,v0)", at=5, zoom=1.3, interactive=1).close()
