"""Fit a plane to regions of a surface defined by
N points that are closest to a given point of the surface."""
from vedo import *


apple = Mesh(dataurl+"apple.ply").subdivide().add_gaussian_noise(0.5)

plt = Plotter()
plt += apple.alpha(0.1)

variances = []
for i, p in enumerate(apple.vertices):
    pts = apple.closest_point(p, n=12) # find the N closest points to p
    plane = fit_plane(pts)             # find the fitting plane
    variances.append(plane.variance)
    if i % 200: continue
    plt += plane
    plt += Points(pts)
    plt += Arrow(plane.center, plane.center+plane.normal/5)

plt += __doc__ + "\nNr. of fits performed: "+str(len(variances))
plt.show().close()

