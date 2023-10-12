"""Interpolate the scalar values from
one Mesh or Points object onto another one"""
from vedo import *
import numpy as np

mesh = Mesh(dataurl+"bunny.obj")

# pick 100 points where we assume that some scalar value is known
# (can be ANY points, not necessarily taken from the mesh)
pts2 = mesh.vertices[:100]
# assume the value is random
scalars = np.random.randint(45,123, 100)
# create a set of points with this scalar values
points = Points(pts2, r=10).cmap('rainbow', scalars)

# interpolate from points onto the mesh, by averaging the 5 closest ones
mesh.interpolate_data_from(points, n=5).cmap('rainbow').add_scalarbar()

show(mesh, points, __doc__, axes=9).close()
