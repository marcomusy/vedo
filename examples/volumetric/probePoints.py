"""
Probe a voxel dataset at specified points
"""
from vtkplotter import *
import numpy as np

vol = load(datadir+"embryo.slc")

pts = np.random.rand(1000, 3)*256

apts = probePoints(vol, pts).pointSize(3)

#print(apts.scalars()) # check the list of point/cell scalars
scals = apts.scalars(0)

printHistogram(scals, minbin=1, horizontal=1, c='g')

show(vol, apts, Text(__doc__), bg='w', axes=8)
