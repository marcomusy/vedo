'''
Voronoi in 3D with Voro++ library.
'''
from vtkplotter import voronoi3D, Points, show
import numpy as np

#from vtkplotter import settings
#settings.voro_path = '/g/sharpeba/software/bin'

N = 2000
nuclei = np.random.rand(N, 3) - (0.5,0.5,0.5)
ncl = Points(nuclei).clean(0.1) # clean makes points evenly spaced
nuclei = ncl.points()

mesh = voronoi3D(nuclei, tol=.001)
#print(len(mesh.info['cells']), mesh.info['volumes'])

pts_inside = mesh.insidePoints(nuclei)
inpts = Points(pts_inside, r=50, c='r', alpha=0.2)

show(mesh, inpts)
