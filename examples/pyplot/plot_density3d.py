"""Density plot from a distribution of points in 3D"""
import numpy as np
from vedo import *

settings.use_depth_peeling = True

n = 3000
p = np.random.normal(7, 0.3, (n,3))
p[:int(n*1/3) ] += [1,0,0]       # shift 1/3 of the points along x by 1
p[ int(n*2/3):] += [1.7,0.4,0.2]

pts = Points(p, alpha=0.5)

vol = pts.density() # density() returns a Volume
vol.cmap('Dark2').alpha([0.1,1])

r = precision(vol.metadata['radius'][0], 2) # retrieve automatic radius value
vol.add_scalarbar3d(title=f'Density (counts in r_s ={r})', italic=1)

show(pts, vol, __doc__, axes=True).close()

