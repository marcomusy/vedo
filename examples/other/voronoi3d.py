'''
Voronoi in 3D with Voro++ library.
'''
from vtkplotter import voronoi3D, Points, show, settings
import numpy as np

#settings.voro_path = '/g/sharpeba/software/bin'

N = 2000
nuclei = np.random.rand(N, 3) - (0.5,0.5,0.5)
ncl = Points(nuclei).clean(0.1) # clean makes points evenly spaced
nuclei = ncl.getPoints()

actor = voronoi3D(nuclei, tol=.001)
#print(len(actor.info['cells']), actor.info['volumes'])

pts_inside = actor.insidePoints(nuclei)
inpts = Points(pts_inside, r=50, c='r', alpha=0.2)

show(actor, inpts)

