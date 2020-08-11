"""Draw the ellipsoid that contains 50% of a cloud of Points,
then check how many points are inside the surface"""
from vedo import pcaEllipsoid, show
import numpy as np

pts = np.random.randn(10000, 3)/1.5*[3,2,1]  # random gaussian point cloud

elli = pcaEllipsoid(pts, pvalue=0.5)

inpcl  = elli.insidePoints(pts).c('green').alpha(0.2)
outpcl = elli.insidePoints(pts, invert=True).c('red').alpha(0.2)

# Extra info can be retrieved with:
print("axis 1 size:", elli.va)
print("axis 2 size:", elli.vb)
print("axis 3 size:", elli.vc)

print("inside  points #", inpcl.NPoints() )
print("outside points #", outpcl.NPoints() )
print("asphericity:", elli.asphericity(), '+-', elli.asphericity_error())

show(elli, inpcl, outpcl, __doc__, axes=1)
