"""
Perform other simple mathematical operation between 3d Volumes.
Possible operations are: +, -, /, 1/x, sin, cos, exp, log, abs, **2, sqrt,
  min, max, atan, atan2, median, mag, dot, gradient, divergence, laplacian.
Alphas defines the opacity transfer function in the scalar range.
"""
print(__doc__)

from vtkplotter import *

vp = Plotter(N=6)

v0 = load(datadir+'embryo.slc').c(0)
vp.show(v0, at=0)

v1 = v0.clone().operation("gradient")
v1 = v1.operation("+", 92.0).c(1).alpha([0, 1, 0, 0, 0])
vp.show(v1, at=1)

v2 = v0.clone().operation("divergence").c(2)
vp.show(v2, at=2)

v3 = v0.clone().operation("laplacian").c(3).alpha([0, 1, 0, 0, 1])
vp.show(v3, at=3)

v4 = v0.clone().operation("median").c(4)
vp.show(v4, at=4)

v5 = v0.clone().operation("dot", v0).c(7)
vp.show(v5, at=5, zoom=1.3, interactive=1)
