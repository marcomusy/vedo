"""Moebius strip with
matplotlib.tri.Triangulation
"""
# https://matplotlib.org/mpl_examples/mplot3d/trisurf3d_demo2.py
import numpy as np
from matplotlib.tri import Triangulation
from vtkplotter import Text2D, Mesh, show

# Make a mesh in the space of parameterisation variables u and v
u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
v = np.linspace(-0.5, 0.5, endpoint=True, num=10)
u, v = np.meshgrid(u, v)
u, v = u.flatten(), v.flatten()

# Mobius mapping, taking a u, v pair and returning x, y, z
x = (1 + 0.5 * v * np.cos(u/2.0)) * np.cos(u)
y = (1 + 0.5 * v * np.cos(u/2.0)) * np.sin(u)
z = 0.5 * v * np.sin(u/2.0)

# Triangulate parameter space to determine the triangle faces
tri = Triangulation(u, v)
points, faces = np.c_[x,y,z], tri.triangles

mesh = Mesh((points, faces), c='orange')
mesh.computeNormals().phong().lineWidth(0.1).lighting('glossy')

show(mesh, Text2D(__doc__), axes=1)
