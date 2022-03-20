"""A simple quiver plot"""
from vedo import Grid, Arrows2D, show


# Create displacements
pts1 = Grid(s=[1.0,1.0]).points()
pts2 = Grid(s=[1.2,1.2]).rotateZ(4).points()

quiv = Arrows2D(pts1, pts2, c="red5")

show(quiv, __doc__, axes=1, zoom=1.2).close()
