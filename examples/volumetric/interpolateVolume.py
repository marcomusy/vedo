"""
Generate a voxel dataset (vtkImageData) by interpolating a scalar
which is only known on a scattered set of points or mesh.
Available interpolation kernels are: shepard, gaussian, voronoi, linear
"""
# Author: Giovanni Dalmasso
from vtkplotter import *
import numpy as np


npts = 60                        # nr. of points of known scalar value
coords = np.random.rand(npts, 3) # range is [0, 1]
scals = np.abs(coords[:, 2])     # let the scalar be the z of point itself

apts = Points(coords).addPointScalars(scals,'scals')

vol = interpolateToVolume(apts, kernel='shepard', radius=0.3)
vol.c(["r", "g", "b"]).alpha([0.4, 0.8])

printHistogram(vol, bins=25, c='b')

write(vol, 'cube.vti')

show(apts, vol, Text(__doc__), bg="white", axes=8)
