"""
In this example we fit spheres to a region of a surface defined by
N points that are closest to a given point of the surface.
For some of these point we show the fitting sphere.
Red lines join the center of the sphere to the surface point.
Blue points are the N points used for fitting.
Green histogram is the distribution of residuals from the fitting.
Red histogram is the distribution of the curvatures (1/r**2).
Fitted radius can be accessed from actor.info['radius'].
"""
from __future__ import division, print_function
from vtkplotter import *

vp = Plotter(verbose=0, axes=0, bg='white')

# load mesh and increase by a lot (N=2) the nr of surface vertices
s = vp.load(datadir+"cow.vtk").alpha(0.3).subdivide(N=2)

reds, invr = [], []
for i, p in enumerate(s.getPoints()):
    if i % 1000:
        continue  # skip most points
    pts = s.closestPoint(p, N=16)  # find the N closest points to p
    sph = fitSphere(pts).alpha(0.05)  # find the fitting sphere
    if sph is None:
        continue  # may fail if all points sit on a plane
    vp += sph
    vp += Points(pts)
    vp += Line(sph.info["center"], p, lw=2)
    reds.append(sph.info["residue"])
    invr.append(1 / sph.info["radius"] ** 2)

vp += histogram(reds, title="residue", bins=12, c="g", pos=3)
vp += histogram(invr, title="1/r**2", bins=12, c="r", pos=4)

vp += Text(__doc__)
vp.show(viewup="z")
