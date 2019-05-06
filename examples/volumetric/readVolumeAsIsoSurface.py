"""
Example to read volumetric data in the form of a tiff stack
or SLC (StereoLithography Contour) from files with automatic isosurfacing:

A tiff stack is a set of image slices in z. The scalar value
(intensity of white) is used to create an isosurface by fixing a threshold.
In this example the level of white is in the range 0=black -> 150=white
If threshold=None this is set to 1/3 of the scalar range.

- Setting connectivity to True discards the small isolated pieces of
surface and only keeps the largest connected surface.

- Smoothing applies a gaussian smoothing with a standard deviation
which is expressed in units of pixels.

- If the spacing of the tiff stack is uneven in xyz, this can be
fixed by setting scaling factors with scaling=[xfac,yfac,zfac]
"""
print(__doc__)
from vtkplotter import show, load, datadir

# Read volume data from a tif file:
f = datadir+"embryo.tif"
a0 = load(f, threshold=80, connectivity=1) # isosurfacing on the fly
a1 = load(f, threshold=80, connectivity=0)
a2 = load(f, smoothing=2)

vp1 = show(a0, a1, a2, shape=(1, 3), axes=1)

# Can also read SLC files
a3 = load(datadir+"embryo.slc", c="g", smoothing=1, connectivity=1)

# newPlotter triggers the instantiation of a new Plotter object
vp2 = show(a3, pos=(300, 300), newPlotter=True)
