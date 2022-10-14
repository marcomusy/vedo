"""Delaunay 3D tetralization"""
from vedo import *
import numpy as np

settings.use_depth_peeling = True

pts = (np.random.rand(10000, 3)-0.5)*2

s = Sphere().alpha(0.1)
pin = s.inside_points(pts)
pin.subsample(0.05)  # impose min separation (5% of bounding box)
printc("# of points inside the sphere:", pin.npoints)

tmesh = delaunay3d(pin).shrink(0.95)

cmesh = tmesh.cut_with_plane(normal=(1,2,-1))

show([(s, pin, "Generate points in a Sphere"),
      (cmesh, __doc__),
     ], N=2, axes=1,
).close()
