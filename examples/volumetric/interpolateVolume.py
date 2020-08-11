"""Generate a Volume by interpolating a scalar
which is only known on a scattered set of points or mesh.
The blue layer is the result of thresholding the volume
between 0.3 and 0.4 and assigning it the new value 0.9 (blue)
"""
from vedo import *
import numpy as np

npts = 500                       # nr. of points of known scalar value
coords = np.random.rand(npts, 3) # range is [0, 1]
scals = np.abs(coords[:, 2])     # let the scalar be the z of the point itself

apts = Points(coords).addPointArray(scals, name='scals')

# Now interpolate these points to a full Volume
# Available interpolation kernels are: shepard, gaussian, voronoi, linear.
vol = interpolateToVolume(apts, kernel='shepard', radius=0.2, dims=(90,90,90))

vol.c(["maroon","g","b"])        # set color   transfer function
vol.alpha([0.3, 0.9])            # set opacity transfer function
#vol.alpha([(0.3,0.3), (0.9,0.9)]) # alternative way, by specifying (xscalar, alpha)
vol.alphaUnit(0.5)               # make the whole object less transparent (defualt is 1)

vol.addScalarBar3D(sy=1, title='height is the scalar').rotateX(90).pos(1.15,1,0.5)

# replace voxels of specific range with a new value
vol.threshold(above=0.3, below=0.4, replaceWith=0.9)

vol.printHistogram(bins=25, c='b')

show(apts, vol, __doc__, axes=1, elevation=-90)
