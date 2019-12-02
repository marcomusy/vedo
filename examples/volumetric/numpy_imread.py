# Create a Volume from a numpy object using imread 
#
# https://github.com/marcomusy/vtkplotter/issues/78
from vtkplotter import *
from skimage.io import imread

f = datadir+'embryo.tif'

voriginal = load(f)
printc('voxel size is', voriginal.spacing(), c='cyan')

raw = imread(f)

vraw = Volume(raw.transpose(2,1,0),
              shape=raw.shape,
              spacing=(104,104,104))

vraw.permuteAxes(2,1,0).mirror("y")

show(voriginal, vraw, N=2, sharecam=0, axes=1)

