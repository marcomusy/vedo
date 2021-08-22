"""Mesh smoothing with two different methods"""
from vedo import Plotter, dataurl

plt = Plotter(N=2)

# Load a mesh and show it
vol = plt.load(dataurl+"embryo.tif")
m0 = vol.isosurface().normalize().lw(0.1).c("violet")
plt.show(m0, __doc__+"\nOriginal Mesh:", at=0)
plt.background([0.8, 1, 1], at=0)  # set first renderer color

# Smooth the mesh
m1 = m0.clone().smooth(niter=20).color("lg")

plt.show(m1, "Polygons are smoothed:", at=1, 
         viewup='z', zoom=1.5, interactive=True).close()