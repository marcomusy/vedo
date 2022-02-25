"""Mesh a line contour with quads of variable resolution"""
from vedo import Spline, Grid, show
import numpy as np

pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.1, 4.0],
        [1.0, 1.5],
        [0.2, 5.0],
        [-1., 3.0],
        [0.4, 2.7],
        [-1., 2.4],
]

shape = Spline(pts, closed=True).color('red4').lineWidth(5)

xcoords = np.arange(-2.0, 2.5, 0.075)
ycoords = np.arange(-0.5, 5.5, 0.075)

xcoords += np.cos(xcoords+0.6)*0.75 # make quads shrink and stretch
ycoords += np.sin(ycoords+0.5)*0.75 # to refine mesh resolution

grd = Grid(s=[xcoords, ycoords])    # create a gridded plane

msh = shape.tomesh(grid=grd, quads=True)

show(shape, msh, __doc__, axes=1).close()

