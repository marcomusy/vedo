"""Histogram along a PCA axis"""
from vedo import *
from vedo.pyplot import histogram

data = np.random.randn(500,3)*[2,1,0.001]
pts = Points(data, r=8, c='#1f77b4').rotateZ(50)  # rotate them!

# Recover the rotation pretending we only know the points
mypts = pts.clone()

# Fit an ellipse (ellipsoid) to the points
elli = pcaEllipsoid(pts).lighting('off')
a1 = Arrow2D([0,0], -3*versor(elli.axis1[:2]))
a2 = Arrow2D([0,0],  3*versor(elli.axis2[:2]))

angle = np.arctan2(elli.axis1[1], elli.axis1[0]) * 57.3
mypts.rotateZ(-angle)  # rotate back to make the histo
histo = histogram(
    mypts.points()[:,1],
    padding=0,
    xlim=(-3,3),
    ytitle='', title=' ',
    c='#1f77b4',
)
histo.origin([0,0,0]).rotateZ(90+angle).pos(2*a1.top)

gon = Goniometer(2*a1.top, [0,0,0], [2*a2.top[0], 0,0])

show(pts.z(-0.2), elli, a1, a2, histo, gon, zoom='tight', axes=8).close()
