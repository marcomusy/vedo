# Example to read volumetric data in the form of a tiff stack 
# or SLC (StereoLithography Contour) files:
#
# A tiff stack is a set of image slices in z. The scalar value 
# (intensity of white) is used to create an isosurface by fixing a threshold.
# In this example the level of white is in the range 0=black -> 150=white
# If threshold=None this is set to 1/3 of the scalar range.
#
# - Setting connectivity to True discards the small isolated pieces of 
# surface and only keeps the largest connected surface.
#
# - Smoothing applies a gaussian smoothing with a standard deviation 
# which is expressed in units of pixels.
#
# - Backface color is set to violet (bc='v') to spot where the vtk  
# reconstruction is (by mistake!) inverting the normals to the surface.
#
# - If the spacing of the tiff stack is uneven in xyz, this can be 
# corrected by setting scaling factors with scaling=[xfac,yfac,zfac]

import vtkplotter

# Read volume data from a tif file:
f = 'data/embryo.tif'

vp = vtkplotter.Plotter(shape=(1,3))
a0= vp.load(f, bc='v', threshold=80, connectivity=1, legend='connectivity=True')
a1= vp.load(f, bc='v', threshold=80, connectivity=0, legend='connectivity=False')
a2= vp.load(f, bc='v', smoothing=2, legend='thres=automatic\nsmoothing=2')

vp.show(a0, at=0)
vp.show(a1, at=1)
vp.show(a2, at=2)

# Can also read SLC files:
vp2= vtkplotter.Plotter(pos=(300,300))
vp2.load('data/embryo.slc', c='g', bc='v', smoothing=1, connectivity=1)
vp2.show()
