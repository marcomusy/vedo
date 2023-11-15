"""Cut a mesh with another mesh"""
from vedo import dataurl, settings, Plotter, Volume, Ellipsoid, show

settings.tiff_orientation_type = 4 # data origin is bottom-left

vol = Volume(dataurl + "embryo.tif")
iso = vol.isosurface(30, flying_edges=False).normalize().pos(0,0,0)

emsh = Ellipsoid().scale(0.4).pos(2.8, 1.5, 1.5).wireframe()

# make a working copy and cut it with the ellipsoid
cut_iso = iso.clone().cut_with_mesh(emsh).c("gold").bc("t")

plt = Plotter(N=2, axes=1)
plt.at(0).show(iso, emsh, __doc__)
plt.at(1).show(cut_iso, viewup="z")
plt.interactive().close()
