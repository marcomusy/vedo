"""Draw the ellipsoid that contains 50% of a cloud of Points,
then check how many points are inside the surface"""
from vedo import *

settings.useDepthPeeling = True

pts = np.random.randn(10000, 3)*[3,2,1] + [50,60,70]

elli = pcaEllipsoid(pts, pvalue=0.50)

a1 = Arrow(elli.center, elli.center + elli.axis1)
a2 = Arrow(elli.center, elli.center + elli.axis2)
a3 = Arrow(elli.center, elli.center + elli.axis3)

inpcl  = elli.insidePoints(pts).c('green',0.2)
outpcl = elli.insidePoints(pts, invert=True).c('red',0.2)

# Extra info can be retrieved with:
print("axis 1 size:", elli.va)
print("axis 2 size:", elli.vb)
print("axis 3 size:", elli.vc)
# print("axis 1 direction:", elli.axis1)
# print("axis 2 direction:", elli.axis2)
# print("axis 3 direction:", elli.axis3)
print("asphericity:", elli.asphericity(), '+-', elli.asphericity_error())

print("inside  points #", inpcl.NPoints() )
print("outside points #", outpcl.NPoints() )

show(elli, a1, a2, a3, inpcl, outpcl, __doc__, axes=1).close()
