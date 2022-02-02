"""Build a volume from a mesh where the
foreground voxels are set to 255 and the background voxels are 0"""
from vedo import Mesh, dataurl, show

surf = Mesh(dataurl+"bunny.obj").normalize().wireframe()

vol = surf.binarize(spacing=(0.02,0.02,0.02))
vol.alpha([0,0.6]).c('blue')

iso = vol.isosurface().color("blue5")

show(vol, surf, __doc__, at=0, N=2, axes=9)
show("..the volume is isosurfaced:", iso, at=1).interactive().close()
