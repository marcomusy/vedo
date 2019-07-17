"""
Using 1D Moving Least Squares to skeletonize a surface.
"""
print(__doc__)

from vtkplotter import *

N = 9    # nr of iterations
f = 0.1  # fraction of neighbours

pts = load(datadir+"man.vtk").decimate(0.1).coordinates()
# pts = load(datadir+'spider.ply').coordinates()
# pts = load(datadir+'magnolia.vtk').subdivide().coordinates()
# pts = load(datadir+'pumpkin.vtk').coordinates()
# pts = load(datadir+'teapot.vtk').coordinates()

a = Points(pts)

for i in range(N):
    show(a, at=i, N=N, elevation=-5)
    a = smoothMLS1D(a.clone(), f).color(i)

interactive()
