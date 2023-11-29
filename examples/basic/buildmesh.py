"""Manually build a mesh from points and faces"""
from vedo import Mesh, show

# Define the vertices and faces that make up the mesh
verts = [(50,50,50), (70,40,50), (50,40,80), (80,70,50)]
cells = [(0,1,2), (2,1,3), (1,0,3)] # cells same as faces

# Build the polygonal Mesh object from the vertices and faces
mesh = Mesh([verts, cells])

# Set the backcolor of the mesh to violet
# and show edges with a linewidth of 2
mesh.backcolor('violet').linecolor('tomato').linewidth(2)

# Create labels for all vertices in the mesh showing their ID
labs = mesh.labels2d('pointid')

# Print the points and faces of the mesh as numpy arrays
print('vertices:', mesh.vertices)
print('faces   :', mesh.cells)

# Show the mesh, vertex labels, and docstring
show(mesh, labs, __doc__, viewup='z', axes=1).close()
