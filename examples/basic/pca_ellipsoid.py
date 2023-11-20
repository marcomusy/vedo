"""Draw the ellipsoid that contains 50% of a cloud of Points,
then check how many points are inside the surface"""
#
# NB: check out pca_ellipse() method for 2D problems
#
from vedo import *

settings.use_depth_peeling = True

pts = Points(np.random.randn(10_000, 3)*[3,2,1])
pts.rotate_z(45).rotate_x(20).shift([30,40,50])

elli = pca_ellipsoid(pts, pvalue=0.50) # 50% of points inside

ids  = elli.inside_points(pts, return_ids=True)
pts.print()  # a new "IsInside" array now exists in pts.pointdata
pin = pts.vertices[ids]
print("inside  points  #", len(pin))

# Create an inverted mask instead of calling inside_points(invert=True)
mask = np.ones(pts.npoints, dtype=bool)
mask[ids] = False
pout = pts.vertices[mask]
print("outside  points #", len(pout))

# Extra info can be retrieved with:
print("axis 1 size:", elli.va)
print("axis 2 size:", elli.vb)
print("axis 3 size:", elli.vc)
print("axis 1 direction:", elli.axis1)
print("axis 2 direction:", elli.axis2)
print("axis 3 direction:", elli.axis3)
print("asphericity:", elli.asphericity(), '+-', elli.asphericity_error())

a1 = Arrow(elli.center, elli.center + elli.axis1)
a2 = Arrow(elli.center, elli.center + elli.axis2)
a3 = Arrow(elli.center, elli.center + elli.axis3)
triad = Assembly(a1, a2, a3) # let's group them

show(
     elli, triad,
     Points(pin).c("green4"),
     Points(pout).c("red5").alpha(0.2),
     __doc__,
     axes=1,
).close()
