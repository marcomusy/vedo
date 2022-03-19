"""Cut a mesh with another mesh"""
from vedo import dataurl, settings, Plotter, Volume, Ellipsoid

settings.tiffOrientationType = 4 # data origin is bottom-left

embryo = Volume(dataurl+"embryo.tif").isosurface(30).normalize()

# mesh used to cut:
msh = Ellipsoid().scale(0.4).pos(2.8, 1.5, 1.5).wireframe()

# make a working copy and cut it with the ellipsoid
cutembryo = embryo.clone().cutWithMesh(msh).c("gold").bc("t")

plt = Plotter(N=2, axes=1)
plt.at(0).show(embryo, msh, viewup="z")
plt.at(1).show(cutembryo, __doc__)
plt.interactive().close()
