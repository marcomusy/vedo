"""A simple quiver plot"""
from vedo import Grid, show
from vedo.pyplot import quiver

# create displacements
pts1 = Grid(sx=1.0, sy=1.0).points()
pts2 = Grid(sx=1.2, sy=1.2).rotateZ(4).points()

qp = quiver(pts1,       # points
            pts2-pts1,  # associated vectors
            c='r',
)

show(qp, __doc__, axes=1).close()
