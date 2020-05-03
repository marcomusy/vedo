"""Read volumetric data from file
with or without automatic isosurfacing"""
from vtkplotter import *

# Read volume data from a tif file:
vol1  = load( datadir+"embryo.tif") # load it as Volume
mesh1 = load( datadir+"embryo.tif", threshold=80) # isosurfacing happens on the fly
plotter1 = show([(vol1, __doc__), mesh1], N=2, axes=8, viewup='z')

# on a new window
vol2 = load(datadir+"embryo.slc") # load it as Volume
vol2.color(['lb','db','dg','dr']) # color transfer values along range
vol2.alpha([0.0, 0.0, 0.2, 0.6, 0.8, 1]) # opacity values along range

# newPlotter triggers the instantiation of a new Plotter object
plotter2 = show(vol2,
                pos=(300, 300),
                viewup='z', zoom=1.5,
                newPlotter=True)
