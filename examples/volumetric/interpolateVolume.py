"""Generate a Volume by interpolating a scalar
which is only known on a scattered set of points or mesh.
The blue layer is the result of thresholding the volume
between 0.3 and 0.4 and assigning it the new value 0.9 (blue)"""
from vedo import *
from vedo.pyplot import cornerHistogram
import numpy as np

npts = 500                       # nr. of points of known scalar value
coords = np.random.rand(npts, 3) # range is [0, 1]
scals = coords[:, 2]             # let the scalar be the z of the point itself

pts = Points(coords)
pts.pointdata["scals"] = scals

# Now interpolate these points to a full Volume
# Available interpolation kernels are: shepard, gaussian, voronoi, linear.
vol = interpolateToVolume(pts, kernel='shepard', radius=0.2, dims=(90,90,90))

vol.c(["maroon","g","b"])        # set color   transfer function
vol.alpha([0.3, 0.9])            # set opacity transfer function
#vol.alpha([(0.3,0.3), (0.9,0.9)]) # alternative way, by specifying (xscalar, alpha)
vol.alphaUnit(0.5)               # make the whole object less transparent (default is 1)

# replace voxels of specific range with a new value
vol.threshold(above=0.3, below=0.4, replace=0.9)#.printHistogram()
# Note that scalar range now has changed (you may want to reapply vol.c().alpha())

ch = cornerHistogram(vol, pos="bottom-left")

vol.addScalarBar3D(sy=1, title='Height is the voxel scalar')
vol.scalarbar.rotateX(90).pos(1.15,1,0.5)

show(pts, vol, ch, __doc__, axes=1, elevation=-90).close()
