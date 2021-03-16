"""Mesh smoothing with two different methods"""
from vedo import Plotter, dataurl

plt = Plotter(N=3)

# Load a mesh and show it
vol = plt.load(dataurl+"embryo.tif")
m0 = vol.isosurface().normalize().lw(0.1).c("violet")
plt.show(m0, __doc__+"\nOriginal Mesh:", at=0)

# Adjust mesh using Laplacian smoothing
m1 = m0.clone().smoothLaplacian(niter=20, relaxfact=0.1, edgeAngle=15, featureAngle=60)
m1.color("pink")
plt.show(m1, "Laplacian smoothing", at=1, viewup='z')

# Adjust mesh using a windowed sinc function interpolation kernel
m2 = m0.clone().smoothWSinc(niter=20, passBand=0.1, edgeAngle=15, featureAngle=60)
m2.color("lg")
plt.show(m2, "WindowSinc smoothing", at=2)

plt.backgroundColor([0.8, 1, 1], at=0)  # set first renderer color

plt.show(zoom=1.4, interactive=True)
