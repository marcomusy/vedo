"""Adds new points to an input point cloud. 
The new points are created in such a way that
all points in any local neighborhood are 
within a target distance of one another. 
"""
from vtkplotter import *
import numpy as np
np.random.seed(3)

npts = 200                       # nr. of points
coords = np.random.rand(npts, 3) # range is [0, 1]
scals = np.abs(coords[:, 2])     # let the scalar be the z of point itself

apts = Points(coords, r=9).addPointScalars(scals, name='scals')

densecloud = densifyCloud(apts, .05, closestN=10, maxIter=1)
print(apts.N(), '->', densecloud.N())

ppp = Points(densecloud.coordinates())
show(apts, densecloud, Text(__doc__), axes=8)