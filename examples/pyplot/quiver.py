"""A simple quiver plot"""
from vedo import Grid, show
from vedo.pyplot import quiver

# create displacements
pts1 = Grid(s=[1.0,1.0]).points()
pts2 = Grid(s=[1.2,1.2]).rotateZ(4).points()

qp = quiver(pts1,       # points
            pts2-pts1,  # associated vectors
            c='red5',
)

show(qp, __doc__, axes=1, zoom=1.2).close()
