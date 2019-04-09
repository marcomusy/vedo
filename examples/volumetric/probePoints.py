"""
Probe a voxel dataset at specified points
"""
from vtkplotter import *
import numpy as np

img = loadImageData(datadir+"embryo.slc")

pts = np.random.rand(1000, 3)*256

apts = probePoints(img, pts).pointSize(3)

scals = apts.scalars(0)

printHistogram(scals, minbin=1, horizontal=1, c='g')

show(img, apts, Text(__doc__), bg='w', axes=8)
