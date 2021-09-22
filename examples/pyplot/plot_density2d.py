"""Density plot from a distribution of points in 2D"""
import numpy as np
from vedo import *

settings.useDepthPeeling = True

n = 10000
p = np.random.normal(0, 0.3, (n,2))
p[:int(n*1/3) ] += [1.0, 0.0] # shift 1/3 of the points along x by 1
p[ int(n*2/3):] += [1.7, 0.4]

# create the point cloud
pts = Points(p).color('k', 0.2)

# radius of local search can be specified (None=automatic)
vol = pts.density(radius=None).c('Paired_r') # returns a Volume

# Other cool color mapping: Set1_r, Dark2. Or you can build your own, e.g.:
# vol.c(['w','w','y','y','r','r','g','g','b','k']).alpha([0,1])

r = precision(vol.info['radius'], 2) # retrieve automatic radius value
vol.addScalarBar3D(title='Density (counts in r_search ='+r+')', c='k', italic=1)

show([(pts,__doc__), vol], N=2, axes=True).close()

