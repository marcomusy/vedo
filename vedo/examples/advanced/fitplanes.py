"""Fit a plane to regions of a surface defined by
N points that are closest to a given point of the surface.
Green histogram is the distribution of residuals from the fitting.
"""
from vedo import *
from vedo.pyplot import histogram

plt = Plotter()

apple = load(datadir+"apple.ply").subdivide().pointGaussNoise(1)
plt += apple.alpha(0.1)

variances = []
for i, p in enumerate(apple.points()):
    pts = apple.closestPoint(p, N=12) # find the N closest points to p
    plane = fitPlane(pts)             # find the fitting plane
    variances.append(plane.variance)
    if i % 400: continue
    plt += plane
    plt += Points(pts)
    plt += Arrow(plane.center, plane.center+plane.normal/5)

plt += histogram(variances, xtitle='variance').scale(6).pos(1.2,.2,-1)
plt += __doc__ + "\nNr. of fits performed: "+str(len(variances))
plt.show()
