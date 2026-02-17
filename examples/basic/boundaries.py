"""Extract points on the boundary of a mesh.
Add an ID label to all vertices."""
from vedo import Mesh, dataurl, Points, show

# Load and pre-process mesh.
b = Mesh(dataurl + "290.vtk")
b.compute_normals().clean().linewidth(0.1)

# Point ids that lie on open boundaries.
pids = b.boundaries(return_point_ids=True)

# Highlight boundary points in red.
pts = Points(b.vertices[pids]).c("red5").ps(10)

labels = b.labels("id", scale=10).c("green2")

show(b, pts, labels, __doc__, zoom=2).close()
