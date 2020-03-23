"""Draw the PCA (Principal Component Analysis) ellipsoid that contains
50% of a cloud of Points, then check if points are inside the surface.
Extra info cab be retrieved with:
mesh.asphericity()
mesh.asphericity_error()
"""
from vtkplotter import Plotter, pcaEllipsoid, Points, Text2D
import numpy as np


vp = Plotter(axes=1)

pts = np.random.randn(500, 3)*[3,2,1]  # random gaussian point cloud

elli = pcaEllipsoid(pts, pvalue=0.5)

ipts = elli.insidePoints(pts)  # elli is a vtkAssembly
opts = elli.insidePoints(pts, invert=True)

vp += Points(ipts, c="g")
vp += Points(opts, c="r")
vp += [elli, Text2D(__doc__)]

print("inside  points #", len(ipts))
print("outside points #", len(opts))
print("asphericity:", elli.asphericity(), '+-', elli.asphericity_error())
print("axis 1 size:", elli.va)
print("axis 2 size:", elli.vb)
print("axis 3 size:", elli.vc)
vp.show()
