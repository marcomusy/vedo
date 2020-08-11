"""
Using 1D Moving Least Squares to skeletonize a surface.
"""
print(__doc__)
from vedo import *

N = 9    # nr of iterations
f = 0.2  # fraction of neighbours

pts = load(datadir+"man.vtk").clean(tol=0.02).points()
# pts = load(datadir+'spider.ply').points()
# pts = load(datadir+'magnolia.vtk').subdivide().points()
# pts = load(datadir+'apple.ply').points()
# pts = load(datadir+'teapot.vtk').points()

pc = Points(pts)

for i in range(N):
    pc = pc.clone().smoothMLS1D(f=f).color(i)
    show(pc, at=i, N=N, elevation=-5)

interactive()
