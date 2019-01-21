# Generate a time sequence of 3D shapes (from a sphere to a tetrahedron)
# as noisy cloud points, and smooth it with Moving Least Squares (smoothMLS3D).
# This make a simultaneus fit in 4D (space+time).
# smoothMLS3D method returns a vtkActor where points are color coded
# in bins of fitted time. 
# Data itself can suggest a meaningful time separation based on the spatial 
# distribution of points.
# The nr neighbours in the local 4D fitting must be specified.
# 
import numpy as np
from vtkplotter import Plotter, sphere, smoothMLS3D


vp = Plotter(N=2, axes=0)

# generate uniform points on sphere (tol separates points by 2% of actor size)
cc = sphere(res=200).clean(tol=0.02).coordinates() 

a, b, noise = .2, .4, .1 # some random warping paramenters, and noise factor
for i in range(5):       # generate a time sequence of 5 shapes
    cs = cc + a * i * cc**2 + b * i * cc**3  # warp sphere in weird ways
    # set absolute time of points actor, and add 1% noise on positions
    vp.points(cs, c=i, alpha=0.5).gaussNoise(1.0).time(0.2*i)
    vp.show(at=0, zoom=1.4)                  # show input clouds as func(time)
    
asse = smoothMLS3D(vp.actors, neighbours=50)

vp.addScalarBar3D(asse, at=1, pos=(-2,0,-1)) # color indicates fitted time
vp.show(asse, at=1, zoom=1.4, axes=4, interactive=1)
