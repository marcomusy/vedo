"""Histogram along a PCA axis"""
import numpy as np
from vedo import Points, pca_ellipse, Arrow2D, Goniometer
from vedo.pyplot import Figure, histogram

data = np.random.randn(1000, 3)

pts = Points(data).color('#1f77b4').ps(6)
pts.scale([2,1,0.01]).rotate_z(45).shift(5,1)  # rotate and shift!

# Recover the rotation pretending we only know the points
# Fit a 1-sigma ellipse to the points
elli = pca_ellipse(pts)

ec, e1, e2 = elli.center, elli.axis1, elli.axis2
arrow1 = Arrow2D(ec, ec - 3*e1)
arrow2 = Arrow2D(ec, ec + 3*e2)

angle = np.arctan2(e1[1], e1[0]) * 180/np.pi
mypts = pts.clone()  # rotate back to make the histo:
mypts.shift(-ec).rotate_z(-angle)
histo = histogram(         # a Histogram1D(Figure) object
    mypts.vertices[:,1],   # grab the y-values (PCA2)
    ytitle='', title=' ',  # no automatic title, no y-axis
    c='#1f77b4',           # color
    aspect=16/9,           # aspect ratio
)
histo.rotate_z(90 + angle).pos(ec - 6*e1)

gon = Goniometer(ec-5.5*e1, ec, [ec[0]-5.5*e1[0], ec[1],0]).z(0.2)

fig = Figure([0,14], [-4,9], aspect="equal", title=__doc__)
fig += [pts, elli.z(-0.1), arrow1, arrow2, gon, histo]
fig.show(zoom='tight').close()
