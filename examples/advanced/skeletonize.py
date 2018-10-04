# Using 1D Moving Least Squares to skeletonize a surface.
#
from __future__ import division, print_function
from vtkplotter import Plotter

N=10  # nr of iterations
f=0.1 # fraction of neighbours 

vp = Plotter(N=N, axes=0)

pts = vp.load('data/shapes/man.vtk').decimate(0.1).coordinates()
#pts = vp.load('data/shapes/spider.ply').coordinates()
#pts = vp.load('data/shapes/magnolia.vtk').subdivide().coordinates()
#pts = vp.load('data/shapes/pumpkin.vtk').coordinates()
#pts = vp.load('data/shapes/teapot.vtk').coordinates()

a = vp.points(pts)
for i in range(N):
    vp.show(a, at=i, legend='iter #'+str(i))
    a = a.clone().color(i)
    vp.smoothMLS1D(a, f)
    
vp.show(interactive=1)












































