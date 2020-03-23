"""
Generate a Volume by interpolating a scalar
which is only known on a scattered set of points or mesh.
Available interpolation kernels are: shepard, gaussian, voronoi, linear.
The blue layer is the result of thresholding the volume
between 0.3 and 0.4 and assigning it the new value 0.9
"""
# Author: Giovanni Dalmasso
from vtkplotter import *
import numpy as np

npts = 500                       # nr. of points of known scalar value
coords = np.random.rand(npts, 3) # range is [0, 1]
scals = np.abs(coords[:, 2])     # let the scalar be the z of point itself

apts = Points(coords).addPointScalars(scals, name='scals')

vol = interpolateToVolume(apts, kernel='shepard', radius=0.2, dims=(90,90,90))
vol.c(["tomato", "g", "b"]).alpha([0.4, 0.8]) # set color/opacity transfer functions

vol.threshold(vmin=0.3, vmax=0.4, replaceWith=0.9) # replace voxel value in [vmin,vmax]

printHistogram(vol, bins=25, c='b')

#write(vol, 'cube.vti')

show(apts, vol, Text2D(__doc__), axes=1, viewup='z')
