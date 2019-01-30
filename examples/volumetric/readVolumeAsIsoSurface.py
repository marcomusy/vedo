'''
Example to read volumetric data in the form of a tiff stack 
or SLC (StereoLithography Contour) files:

A tiff stack is a set of image slices in z. The scalar value 
(intensity of white) is used to create an isosurface by fixing a threshold.
In this example the level of white is in the range 0=black -> 150=white
If threshold=None this is set to 1/3 of the scalar range.

- Setting connectivity to True discards the small isolated pieces of 
surface and only keeps the largest connected surface.

- Smoothing applies a gaussian smoothing with a standard deviation 
which is expressed in units of pixels.

- If the spacing of the tiff stack is uneven in xyz, this can be 
corrected by setting scaling factors with scaling=[xfac,yfac,zfac]
'''
from vtkplotter import Plotter, load, text

# Read volume data from a tif file:
f = 'data/embryo.tif'

vp = Plotter(shape=(1,4), axes=0)

vp.show(text(__doc__), at=3)

a0 = load(f, threshold=80, connectivity=1, legend='connectivity=True')
a1 = load(f, threshold=80, connectivity=0, legend='connectivity=False')
a2 = load(f, smoothing=2, legend='thres=automatic\nsmoothing=2')

vp.show(a0, at=0)
vp.show(a1, at=1)
vp.show(a2, at=2)


#### Can also read SLC files 
#(NB: vp2.load instead of load. This appends the new actor in vp2.actors):
vp2= Plotter(pos=(300,300))
vp2.load('data/embryo.slc', c='g', smoothing=1, connectivity=1)
vp2.show()
