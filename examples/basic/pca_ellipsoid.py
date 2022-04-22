from vedo import *

settings.useDepthPeeling = True

pts = Points(np.random.randn(10000, 3)*[3,2,1] + [50,60,70])

elli = pcaEllipsoid(pts, pvalue=0.50)

elli.insidePoints(pts, returnIds=True)

ids  = elli.insidePoints(pts, returnIds=True)
pts.print()  # a new "IsInside" array now exists in pts
pin = pts.points()[ids]
print("inside  points #", len(pin))

# Create an inverted mask instead of calling insidePoints(invert=True)
mask = np.ones(pts.N(), dtype=bool)
mask[ids] = False
pout = pts.points()[mask]
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

show(elli,
     a1, a2, a3,
     Points(pin).c("green4"),
     Points(pout).c("red5").alpha(0.2),
     axes=1,
).close()
