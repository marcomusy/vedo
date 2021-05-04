"""Represent a volume as lego blocks (voxels).
Colors correspond to the volume's scalar.
Try also:
> vedo --lego data/embryo.tif"""
# https://matplotlib.org/users/colormaps.html
from vedo import *

vol = Volume(dataurl+'embryo.tif').printHistogram(logscale=True)

vol.crop(back=0.5) # crop 50% from neg. y

# show lego blocks whose value is between vmin and vmax
lego = vol.legosurface(vmin=60, cmap='seismic')

# make colormap start at 40
lego.addScalarBar(horizontal=True, c='k')

show(lego, __doc__, axes=1, viewup='z').close()
