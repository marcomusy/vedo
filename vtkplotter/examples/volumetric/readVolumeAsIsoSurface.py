"""Read volumetric data from file
with or without automatic isosurfacing
"""
from vtkplotter import *

comment = Text2D(__doc__)

# Read volume data from a tif file:
f = datadir+"embryo.tif"
vol1  = load(f) # load it as Volume
mesh1 = load(f, threshold=80) # isosurfacing happens on the fly
plotter1 = show([(vol1, comment), mesh1], N=2, axes=8, viewup='z')

# on a new window
fpath = download("https://vtkplotter.embl.es/data/embryo.slc")
vol2 = load(fpath) # load it as Volume
vol2.color(['lb','db','dg','dr']) # color transfer values along range
vol2.alpha([0.0, 0.0, 0.2, 0.6, 0.8, 1]) # opacity values along range

# newPlotter triggers the instantiation of a new Plotter object
plotter2 = show(vol2, pos=(300, 300),
                viewup='z', zoom=1.5,
                newPlotter=True)
