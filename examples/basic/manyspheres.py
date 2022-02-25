"""Example that shows how to draw very large number
of spheres (same for Points, Lines) with different
colors or different radii, N="""
from vedo import show, Spheres
from random import gauss

N = 50000

cols = range(N)  # color numbers
pts = [(gauss(0, 1), gauss(0, 2), gauss(0, 1)) for i in cols]
rads = [abs(pts[i][1]) / 10 for i in cols]  # radius=0 for y=0

# all have same radius but different colors:
s0 = Spheres(pts, c=cols, r=0.1, res=5)  # res= theta-phi resolution
show(s0, __doc__+str(N), at=0, N=2, axes=1, viewup=(-0.7, 0.7, 0))

# all have same color but different radius along y:
s1 = Spheres(pts, r=rads, c="lb", res=8)
show(s1, at=1, axes=2).interactive().close()
