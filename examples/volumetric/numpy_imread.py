# Create a Volume from a numpy object using imread
#
# https://github.com/marcomusy/vtkplotter/issues/78
from vtkplotter import *
from skimage.io import imread

f = datadir+'embryo.tif'

voriginal = load(f)
printc('voxel size is', voriginal.spacing(), c='cyan')

raw = imread(f)

vraw = Volume(raw, spacing=(104,104,104))
# Need to change axes and mirror
# NOTE: spacing specified above is now inverted: (z,y,x)
vraw.permuteAxes(2,1,0).mirror("y")

# Compare loading the volume directly with the numpy volume:
# they should be the same
show(voriginal, vraw, N=2, axes=1)

