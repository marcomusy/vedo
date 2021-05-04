"""Fit spheres to a region of a surface defined by
N points that are closest to a given point of the surface.
For some of these point we show the fitting sphere.
Red lines join the center of the sphere to the surface point.
Blue points are the N points used for fitting"""
from vedo import *

settings.defaultFont = 'Kanopus'

plt = Plotter()

# load mesh and increase by a lot (N=2) the nr of surface vertices
s = plt.load(dataurl+"cow.vtk").alpha(0.3).subdivide(N=2)

for i, p in enumerate(s.points()):
    if i % 1000:
        continue  # skip most points
    pts = s.closestPoint(p, N=16)  # find the N closest points to p
    sph = fitSphere(pts).alpha(0.05)  # find the fitting sphere
    if sph is None:
        continue  # may fail if all points sit on a plane
    plt += sph
    plt += Points(pts)
    plt += Line(sph.center, p, lw=2)

plt += __doc__
plt.show(viewup="z", axes=1).close()
