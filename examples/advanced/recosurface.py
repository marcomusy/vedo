# Example to show how to use recoSurface() to reconstruct a surface from points.
# 1. An object is loaded and noise is added to its vertices.
# 2. the point cloud is smoothened with MLS (see moving_least_squares.py)
# 3. clean(actor) imposes a minimum distance among mesh points where 
#    'tol' is the fraction of the actor size.
# 4. a triangular mesh is extracted from this set of sparse points
#    'bins' is the number of voxels of the subdivision
# NB: recoSurface only works with vtk version >7
# 
from __future__ import division, print_function
from vtkplotter import Plotter
from vtkplotter.analysis import recoSurface, smoothMLS2D
from vtkplotter.utils import clean
import numpy as np


vp = Plotter(shape=(1,4), axes=0)

act = vp.load('data/shapes/pumpkin.vtk')
vp.show(act, at=0)

noise = np.random.randn(act.N(), 3)*0.05

act_pts0 = vp.points(act.coordinates()+noise, r=3) #add noise
act_pts1 = act_pts0.clone()   #make a copy to modify
vp.show(act_pts0, at=1, legend='noisy cloud')

smoothMLS2D(act_pts1, f=0.4)  #smooth cloud, input actor is modified 

print('Nr of points before cleaning polydata:', act_pts1.N())
clean(act_pts1, tol=0.01) #impose a min distance among mesh points
print('             after  cleaning polydata:', act_pts1.N())

vp.show(act_pts1, at=2, legend='smooth cloud')

act_reco = recoSurface(act_pts1, bins=128) #reconstructed from points
vp.show(act_reco, at=3, ruler=1, interactive=1, legend='surf reco')














































