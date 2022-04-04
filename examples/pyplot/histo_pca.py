"""Histogram along a PCA axis"""
import numpy as np
from vedo import Points, pcaEllipsoid, Arrow2D, Goniometer
from vedo.pyplot import Figure, histogram
# np.random.seed(2)
data = np.random.randn(1000, 3)

pts = Points(data, r=6, c='#1f77b4')
pts.scale([2,1,0.01]).rotateZ(45).shift(5,1)  # rotate and shift!

# Recover the rotation pretending we only know the points
# Fit an ellipse to the points
# elli = pcaEllipsoid(pts).lighting('off')
elli = pcaEllipsoid(pts)
ec, e1, e2 = elli.center, elli.axis1, elli.axis2
arrow1 = Arrow2D(ec, ec - 3*e1)
arrow2 = Arrow2D(ec, ec + 3*e2)

angle = np.arctan2(e1[1], e1[0]) * 180/np.pi
mypts = pts.clone()  # rotate back to make the histo:
mypts.shift(-ec).rotateZ(-angle)
histo = histogram(         # a Histogram1D(Figure) object
    mypts.points()[:,1],   # grab the y-values (PCA2)
    ytitle='', title=' ',  # no automatic title, no y-axis
    c='#1f77b4',           # color
    aspect=16/9,           # aspect ratio
)
histo.rotateZ(90 + angle).pos(ec - 6*e1)

gon = Goniometer(ec-5.5*e1, ec, [ec[0]-5.5*e1[0], ec[1],0]).z(0.2)

fig = Figure([0,14], [-4,9], aspect="equal", title=__doc__)
fig += [pts, elli, arrow1, arrow2, gon, histo]
fig.show(zoom='tight').close()
