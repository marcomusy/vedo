"""Manually build a mesh from points and faces"""
from vedo import Mesh, show

# Vertices and triangular faces (indices into verts).
verts = [(50, 50, 50), (70, 40, 50), (50, 40, 80), (80, 70, 50)]
cells = [(0, 1, 2), (2, 1, 3), (1, 0, 3)]

mesh = Mesh([verts, cells])

# Show both front and back styling to make topology easy to read.
mesh.backcolor("violet").linecolor("tomato").linewidth(2)

labs = mesh.labels2d("pointid")

print("vertices:", mesh.vertices)  # same as mesh.points or mesh.coordinates
print("faces   :", mesh.cells)

show(mesh, labs, __doc__, viewup="z", axes=1).close()
