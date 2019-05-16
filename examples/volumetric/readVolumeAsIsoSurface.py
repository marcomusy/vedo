"""
Example to read volumetric data in the form of a tiff stack
or SLC (StereoLithography Contour) from files
with or without automatic isosurfacing:

A tiff stack is a set of image slices in z. The scalar value
(intensity of white) is used to create an isosurface by fixing a threshold.
If threshold=None this is set to 1/3 of the scalar range.

- If the spacing of the tiff stack is uneven in xyz, this can be
fixed by setting scaling factors with scaling=[xfac,yfac,zfac]
"""
print(__doc__)
from vtkplotter import show, load, datadir

# Read volume data from a tif file:
f = datadir+"embryo.tif"

v = load(f) # Volume
a = load(f, threshold=80) # isosurfacing on the fly
vp1 = show(v, a, shape=(1, 2), axes=8, viewup='z')

# Can also read SLC files
vol = load(datadir+"embryo.slc") # Volume
vol.color(['lb','db','dg','dr']) # color transfer values along range
vol.alpha([0.0, 0.0, 0.2, 0.6, 0.8, 1]) # opacity values along range

# newPlotter triggers the instantiation of a new Plotter object
vp2 = show(vol, pos=(300, 300), bg='white',
           viewup='z', zoom=1.5,
           newPlotter=True)
