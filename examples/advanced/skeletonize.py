"""
Using 1D Moving Least Squares to skeletonize a surface.
"""
print(__doc__)
from vedo import *

N = 9    # nr of iterations
f = 0.2  # fraction of neighbours

pts = load(dataurl+"man.vtk").clean(tol=0.02).points()
# pts = load(dataurl+'spider.ply').points()
# pts = load(dataurl+'magnolia.vtk').subdivide().points()
# pts = load(dataurl+'apple.ply').points()
# pts = load(dataurl+'teapot.vtk').points()

pc = Points(pts)

for i in range(N):
    pc = pc.clone().smoothMLS1D(f=f).color(i)
    show(pc, at=i, N=N, elevation=-5)

interactive()
