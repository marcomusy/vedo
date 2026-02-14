"""Represent a volume as lego blocks (voxels).
Colors correspond to the volume's scalar.
Try also:
> vedo --lego data/embryo.tif"""
from vedo import *

vol = Volume(dataurl+'embryo.tif')

vol.crop(back=0.50) # crop 50% from neg. y

# Keep only voxels in value range and display them as lego blocks.
lego = vol.legosurface(vmin=20, vmax=None, boundary=False)
lego.cmap('seismic', vmin=0, vmax=127).add_scalarbar()

show(lego, __doc__, axes=1, viewup='z').close()
