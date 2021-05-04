"""Interpolate the scalar values from
one Mesh or Points object onto another one"""
from vedo import *
import numpy as np

mesh = Mesh(dataurl+"bunny.obj")

# pick 100 points where we assume that some scalar value is known
# (can be ANY points, not necessarily taken from the mesh)
pts2 = mesh.points()[:100]
# assume the value is random
scalars = np.random.randint(45,123, 100)
# create a set fo points with this scalar values
points = Points(pts2, r=10).cmap('rainbow', scalars)

# interpolate from points onto the mesh, by averaging the 5 closest ones
mesh.interpolateDataFrom(points, N=5).cmap('rainbow').addScalarBar()

show(mesh, points, __doc__, axes=9).close()
