"""
Example to show how to use recoSurface()
to reconstruct a surface from points.

 1. An object is loaded and
    noise is added to its vertices.
 2. the point cloud is smoothened with MLS
    (see moving_least_squares.py)
 3. clean(actor) imposes a minimum distance
    among mesh points where 'tol' is the
    fraction of the actor size.
 4. a triangular mesh is extracted from
    this set of sparse Points, 'bins' is the
    number of voxels of the subdivision
"""
print(__doc__)
from vtkplotter import *
import numpy as np


vp = Plotter(N=4, axes=0, bg='w')

act = vp.load(datadir+"pumpkin.vtk")
vp.show(act, at=0)

noise = np.random.randn(act.N(), 3) * 0.04

pts0 = Points(act.coordinates() + noise, r=3).legend("noisy cloud")
vp.show(pts0, at=1)

pts1 = smoothMLS2D(pts0, f=0.4)  # smooth cloud, input actor is modified

print("Nr of points before cleaning polydata:", pts1.N())

# impose a min distance among mesh points
pts1.clean(tol=0.01).legend("smooth cloud")
print("             after  cleaning polydata:", pts1.N())

vp.show(pts1, at=2)

# reconstructed surface from point cloud
reco = recoSurface(pts1, bins=128).legend("surf reco")
vp.show(reco, at=3, axes=7, zoom=1.2, interactive=1)
