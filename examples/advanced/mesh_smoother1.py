"""Compare a mesh before and after iterative smoothing."""
from vedo import dataurl, Plotter, Volume


# Build mesh from an isosurface.
vol = Volume(dataurl + "embryo.tif")
m0 = vol.isosurface(flying_edges=False).normalize()
m0.lw(1).c("violet")

# Smooth the mesh
m1 = m0.clone().smooth(niter=20)
m1.color("lg")

plt = Plotter(N=2)
plt.at(0).background("light blue")  # set first renderer color
plt.show(m0, "Original Mesh:")

plt.at(1)
plt.show("Mesh polygons are smoothed:", m1, viewup="z", zoom=1.5)
plt.interactive().close()
