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
from __future__ import division, print_function
from vtkplotter import *
import numpy as np


vp = Plotter(shape=(1, 5), axes=0, bg='w')
vp.show(Text(__doc__), at=4)

act = vp.load(datadir+"shapes/pumpkin.vtk")
vp.show(act, at=0)

noise = np.random.randn(act.N(), 3) * 0.05

act_pts0 = Points(act.coordinates() + noise, r=3).legend("noisy cloud")
act_pts1 = act_pts0.clone()  # make a copy to modify
vp.show(act_pts0, at=1)

smoothMLS2D(act_pts1, f=0.4)  # smooth cloud, input actor is modified

print("Nr of points before cleaning polydata:", act_pts1.N())

# impose a min distance among mesh points
act_pts1.clean(tol=0.01).legend("smooth cloud")
print("             after  cleaning polydata:", act_pts1.N())

vp.show(act_pts1, at=2)

# reconstructed surface from point cloud
act_reco = recoSurface(act_pts1, bins=128).legend("surf reco")
vp.show(act_reco, at=3, axes=7, interactive=1)
