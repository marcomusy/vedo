'''
Draw the PCA (Principal Component Analysis) ellipsoid that contains  
50% of a cloud of Points, then check if points are inside the surface.
Extra info is stored in actor.info['sphericity'], 'va', 'vb', 'vc'.
'''
from vtkplotter import Plotter, pcaEllipsoid, Points, Text
import numpy as np


vp = Plotter(verbose=0, axes=4)

pts = np.random.randn(500, 3) # random gaussian point cloud

act = pcaEllipsoid(pts, pvalue=0.5, pcaAxes=1)

ipts = act.getActor(0).insidePoints(pts) # act is a vtkAssembly
opts = act.getActor(0).insidePoints(pts, invert=True)
vp.add(Points(ipts, c='g'))
vp.add(Points(opts, c='r'))

print('inside  points #', len(ipts))
print('outside points #', len(opts))
print('sphericity :', act.info['sphericity'])

vp.add([act,  Text(__doc__)])

vp.show()