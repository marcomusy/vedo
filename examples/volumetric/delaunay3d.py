"""Delaunay 3D tetralization"""
from vedo import *
import numpy as np

settings.useDepthPeeling = True

pts = (np.random.rand(10000, 3)-0.5)*2

s = Sphere().alpha(0.1)
pin = s.insidePoints(pts)
pin.subsample(0.05)  # impose min separation (5% of bounding box)
printc("# of points inside the sphere:", pin.N())

tmesh = delaunay3D(pin).shrink(0.95)

cmesh = tmesh.cutWithPlane(normal=(1,2,-1))

show([(s, pin, "Generate points in a Sphere"),
      (cmesh, __doc__),
     ], N=2, axes=1,
).close()
