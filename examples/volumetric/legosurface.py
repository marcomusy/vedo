"""Represent a volume as lego blocks (voxels).
Colors correspond to the volume's scalar.
Try also:
> vedo --lego data/embryo.tif"""
from vedo import *

vol = Volume(dataurl+'embryo.tif')

vol.crop(back=0.50) # crop 50% from neg. y

# show lego blocks whose value is between vmin and vmax
lego = vol.legosurface(vmin=20, vmax=None, boundary=False)
lego.cmap('seismic', on='cells', vmin=0, vmax=127).addScalarBar()

show(lego, __doc__, axes=1, viewup='z').close()
