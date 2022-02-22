"""Generate a denser point cloud.
The new points are created in such a way that
all points in any local neighborhood are
within a target distance of one another"""
from vedo import Points, printc, show
import numpy as np

npts = 50                        # nr. of points
coords = np.random.rand(npts, 3) # range is [0, 1]
scals = np.abs(coords[:, 1])     # let the scalar be the y of the point itself
pts = Points(coords, r=9)
pts.pointdata["scals"] = scals

densecloud = pts.densify(0.1, nclosest=10, niter=1) # return a new pointcloud.Points
printc('nr. points increased', pts.N(), '\rightarrow ', densecloud.N(), c='lg')

show([(pts, __doc__), densecloud], N=2, axes=1).close()
