"""A RectilinearGrid is a dataset where edges are parallel to the coordinate axes.
It can be thought of as a tessellation of a box in 3D space, similar to a Volume
except that the cells are not necessarily cubes, but they can have different
lengths along each axis."""

from vedo import *

xcoords = 7 + np.sqrt(np.arange(0,900,25))
ycoords = np.arange(0, 20)
zcoords = np.arange(0, 20)

rgrid = RectilinearGrid([xcoords, ycoords, zcoords])

print(rgrid)
# print(rgrid.x_coordinates().shape)
# print(rgrid.has_blank_points())
# print(rgrid.compute_structured_coords([20,10,11]))

msh = rgrid.tomesh().lw(1)
show(msh, __doc__, axes=1, viewup="z")
