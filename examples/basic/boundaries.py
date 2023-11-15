"""Extract points on the boundary of a mesh.
Add an ID label to all vertices."""
from vedo import *

# Load a mesh from a URL, compute normals, and clean duplicate points
b = Mesh(dataurl+'290.vtk')
b.compute_normals().clean().linewidth(0.1)

# Get the point IDs on the boundary of the mesh
pids = b.boundaries(return_point_ids=True)

# Create a Points object to represent the boundary points
pts = Points(b.vertices[pids]).c('red5').ps(10)

# Create a Label object for all the vertices in the mesh
labels = b.labels('id', scale=10).c('green2')

# Show the mesh, boundary points, vertex labels, and docstring 
show(b, pts, labels, __doc__, zoom=2).close()
