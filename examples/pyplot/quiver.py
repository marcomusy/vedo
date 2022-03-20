"""A simple quiver plot"""
from vedo import Grid, Arrows2D, show


# create displacements
pts1 = Grid(s=[1.0,1.0]).points()
pts2 = Grid(s=[1.2,1.2]).rotateZ(4).points()

arrs2d = Arrows2D(pts1, pts2, c="red5",
    # shaftLength=shaftLength,
    # shaftWidth=shaftWidth,
    # headLength=headLength,
    # headWidth=0.01,
    s=0.1
)


show(arrs2d, __doc__, axes=1, zoom=1.2).close()
