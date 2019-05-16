"""
Perform other simple mathematical operation between 3d Volumes.
Possible operations are: +, -, /, 1/x, sin, cos, exp, log, abs, **2, sqrt,
  min, max, atan, atan2, median, mag, dot, gradient, divergence, laplacian.
Alphas defines the opacity transfer function in the scalar range.
"""
print(__doc__)

from vtkplotter import Plotter, load, volumeOperation, datadir

vp = Plotter(N=8, axes=0, bg="w")

v0 = load(datadir+"embryo.slc").c(0)  # vtkVolume
vp.show(v0, at=0)

v1 = volumeOperation(v0, "gradient")
v1 = volumeOperation(v1, "+", 92.0).c(1).alpha([0, 1, 0, 0, 0])
vp.show(v1, at=1)

v2 = volumeOperation(v0, "divergence").c(2)
vp.show(v2, at=2)

v3 = volumeOperation(v0, "laplacian").c(3).alpha([0, 1, 0, 0, 1])
vp.show(v3, at=3)

v4 = volumeOperation(v0, "median").c(4)
vp.show(v4, at=4)

v5 = volumeOperation(v0, "sqrt").c(5)
vp.show(v5, at=5)

v6 = volumeOperation(v0, "log").c(6)
vp.show(v6, at=6)

v7 = volumeOperation(v0, "dot", v0).c(7)
vp.show(v7, at=7, zoom=1.3, interactive=1)
