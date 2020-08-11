"""Mesh smoothing with two different methods"""
from vedo import Plotter, datadir

vp = Plotter(N=3)

# Load a mesh and show it
vol = vp.load(datadir+"embryo.tif")
m0 = vol.isosurface().normalize().lw(0.1).c("violet")
vp.show(m0, __doc__, at=0)

# Adjust mesh using Laplacian smoothing
m1 = m0.clone().smoothLaplacian(niter=20, relaxfact=0.1, edgeAngle=15, featureAngle=60)
m1.color("pink").legend("laplacian")
vp.show(m1, at=1)

# Adjust mesh using a windowed sinc function interpolation kernel
m2 = m0.clone().smoothWSinc(niter=15, passBand=0.1, edgeAngle=15, featureAngle=60)
m2.color("lg").legend("window sinc")
vp.show(m2, at=2)

vp.backgroundColor([0.8, 1, 1], at=0)  # set first renderer color

vp.show(zoom=1.4, interactive=True)
