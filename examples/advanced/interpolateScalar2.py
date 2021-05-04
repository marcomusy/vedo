"""Use scipy to interpolate the value of a scalar known on a set
of points on a new set of points where the scalar is not defined.

Two interpolation methods are possible:
Radial Basis Function (used here), and Nearest Point."""
import numpy as np
from vedo import *
from scipy.interpolate import Rbf, NearestNDInterpolator as Near

mesh = Mesh(dataurl+"bunny.obj").normalize()
pts = mesh.points()

# pick a subset of 100 points where a scalar descriptor is known
ptsubset = pts[:100]

# assume the descriptor value is some function of the point coord y
x, y, z = np.split(ptsubset, 3, axis=1)
desc = 3*sin(4*y)

# build the interpolator to determine the scalar value
#  for the rest of mesh vertices:
itr = Rbf(x, y, z, desc)          # Radial Basis Function interpolator
#itr = Near(ptsubset, desc)       # Nearest-neighbour interpolator

# interpolate desciptor on the full set of mesh vertices
xi, yi, zi = np.split(pts, 3, axis=1)
interpolated_desc = itr(xi, yi, zi)

mesh.cmap('rainbow', interpolated_desc).addScalarBar(title='3sin(4y)')
rpts = Points(ptsubset, r=8, c='white')

show(mesh, rpts, __doc__, axes=1).close()
