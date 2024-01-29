"""Build a volume from a mesh where the
foreground voxels are set to 255 and the background voxels are 0"""
from vedo import Mesh, dataurl, Plotter

surf = Mesh(dataurl+"bunny.obj").normalize().wireframe()

vol = surf.binarize()
vol.alpha([0,0.75]).cmap('blue5')

iso = vol.isosurface().color("blue5")

plt = Plotter(N=2, axes=9)
plt.at(0).show(vol, surf, __doc__)
plt.at(1).show("..the volume is isosurfaced:", iso)
plt.interactive().close()
