"""
Probe a voxel dataset at specified points
and plot a histogram of the values
"""
from vtkplotter import *
import numpy as np

fpath = download("https://vtkplotter.embl.es/data/embryo.slc")
vol = load(fpath)

pts = np.random.rand(2000, 3)*256

mpts = probePoints(vol, pts).pointSize(3)

scals = mpts.getPointArray() # the list of scalars

h = histogram(scals, xlim=(5,120), xtitle='voxel value')
h.scale(2.2)

show(vol, mpts, h, Text2D(__doc__))
