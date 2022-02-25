"""Fit spheres to a region of a surface defined by
N points that are closest to a given point of the surface.
For some of these point we show the fitting sphere.
Red lines join the center of the sphere to the surface point.
Blue points are the N points used for fitting"""
from vedo import *

settings.defaultFont = 'Kanopus'
settings.useDepthPeeling = True

plt = Plotter()

# load mesh and increase by a lot (N=2) the nr of surface vertices
cow = Mesh(dataurl+"cow.vtk").alpha(0.3).subdivide(N=2)

for i, p in enumerate(cow.points()):
    if i % 1000:
        continue  # skip most points
    pts = cow.closestPoint(p, N=16)  # find the N closest points to p
    sph = fitSphere(pts).alpha(0.05)  # find the fitting sphere
    if sph is None:
        continue  # may fail if all points sit on a plane
    plt += sph
    plt += Points(pts)
    plt += Line(sph.center, p, lw=2)

plt += [cow, __doc__]
plt.show(viewup="z", axes=1).close()
