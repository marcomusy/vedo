"""Density plot from a distribution of points in 3D"""
import numpy as np
from vedo import *

settings.useDepthPeeling = True

n = 3000
p = np.random.normal(7, 0.3, (n,3))
p[:int(n*1/3) ] += [1,0,0]       # shift 1/3 of the points along x by 1
p[ int(n*2/3):] += [1.7,0.4,0.2]

pts = Points(p, alpha=0.5)

vol = pts.density().c('Dark2').alpha([0.1,1]) # density() returns a Volume

r = precision(vol.info['radius'], 2) # retrieve automatic radius value
vol.addScalarBar3D(title='Density (counts in r_s ='+r+')', c='k', italic=1)

show(pts, vol, __doc__, axes=True).close()

