# Example to show how to use recoSurface() for surface reconstruction.
# 1. A mesh is loaded and noise is added to its vertices.
# 2. the point cloud is smoothened with MLS (see moving_least_squares.py)
# 3. a triangular mesh is extracted from this set of points
#    neighbors = number of neighbor points to evaluate normals (~20 to ~200)
#    spacing = mesh resolution (~0.2), zero=automatic
#
from __future__ import division, print_function
from plotter import vtkPlotter
import numpy as np


vp = vtkPlotter(shape=(1,4), axes=0)

act = vp.load('data/shapes/pumpkin.vtk', alpha=1)
vp.show(act, at=0)

noise = np.random.randn(act.N(), 3)*.04

act_pts0 = vp.points(act.coordinates()+noise, r=3) #add noise
act_pts1 = act_pts0.clone()
vp.show(act_pts0, at=1, legend='noisy cloud')

vp.smoothMLS(act_pts1, f=0.4) #smooth cloud
vp.show(act_pts1, at=2, legend='smooth cloud')

act_reco = vp.recoSurface(act_pts1, bins=128) #reco
vp.show(act_reco, at=3, interactive=1, legend='surf reco')














































