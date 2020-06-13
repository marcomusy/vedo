"""Create a Volume
from a numpy object using imread"""
#
# https://github.com/marcomusy/vedo/issues/78
from vedo import *
from skimage.io import imread

f = datadir+'embryo.tif'

voriginal = load(f)
printc('voxel size is', voriginal.spacing(), c='cyan')

raw = imread(f)

vraw = Volume(raw, spacing=(104,104,104))
vraw.mirror("y")

# Compare loading the volume directly with the numpy volume:
# they should be the same
show([(voriginal,__doc__), (vraw,"From imread")], N=2, axes=1)
