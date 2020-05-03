"""A simple quiver plot"""
from vtkplotter import Grid, show
from vtkplotter.pyplot import quiver

# create displacements
pts1 = Grid(sx=1.0, sy=1.0).points()
pts2 = Grid(sx=1.2, sy=1.2).rotateZ(4).points()

qp = quiver(pts1,       # points
            pts2-pts1,  # associated vectors
            cmap='jet', # can also be a fixed color
           )

show(qp, __doc__, axes=1)
