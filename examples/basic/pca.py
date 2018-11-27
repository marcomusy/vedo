# Draw the PCA (Principal Component Analysis) ellipsoid that contains 50% of 
# a cloud of points, then check if points are inside the surface.
# Extra info is stored in actor.info['sphericity'], 'va', 'vb', 'vc'.
#
from vtkplotter import Plotter
from vtkplotter.analysis import pca
import numpy as np


vp = Plotter(verbose=0, axes=4)

pts = np.random.randn(200, 3) # random gaussian point cloud

act = pca(pts, pvalue=0.5, pcaAxes=1, legend='PCA ellipsoid')

ipts = act.getActor(0).insidePoints(pts) # act is a vtkAssembly
opts = act.getActor(0).insidePoints(pts, invert=True)
vp.points(ipts, c='g')
vp.points(opts, c='r')

print('in  points #', len(ipts))
print('out points #', len(opts))
print('sphericity :', act.info['sphericity'])

vp.actors.append(act) # add actor to the list to be shown (not automatic)
vp.show()