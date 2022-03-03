from vedo import Plotter, dataurl

plt = Plotter(N=2)

# Load a mesh and show it
vol = plt.load(dataurl+"embryo.tif")
m0 = vol.isosurface().normalize().lw(0.1).c("violet")

# Smooth the mesh
m1 = m0.clone().smooth(niter=20).color("lg")

plt.at(0).show(m0, "Original Mesh:")
plt.background('light blue')  # set first renderer color

plt.at(1).show(
    "Mesh polygons are smoothed:", m1,
    viewup='z', zoom=1.5)
plt.interactive().close()
