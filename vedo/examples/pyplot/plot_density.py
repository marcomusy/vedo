"""Make a density plot from a distribution of points in 3D
(r is the radius of the local search, which can be specified)"""
import numpy as np
from vedo import *

n = 3000
p = np.random.normal(7, 0.3, (n,3))
p[:int(n*1/3) ] += [1,0,0]       # shift 1/3 of the points along x by 1
p[ int(n*2/3):] += [1.7,0.4,0.2]

pts = Points(p, alpha=0.5)

vol = pointDensity(pts, dims=50) # ,radius=0.1)
vol.mode(1).c('Paired_r')        # max projection mode, colormap

# Other cool color mapping: Set1_r, Dark2, or you can build your own:
# vol.c(['w','w','y','y','r','r','g','g','b','k']).alpha([0,1])

r = precision(vol.info['radius'], 2) # retrieve automatic radius value
vol.addScalarBar3D(title='Density (counts in r='+r+')', c='k', italic=1)

show([(pts,__doc__), vol], N=2, axes=True)

