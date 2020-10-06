'''
Mesh objects can be combined with
(1) `mesh.merge` - creates a new mesh object; this new mesh inherits properties (color, etc.) of the first mesh.
(2) `assembly.Assembly` - combines meshes (or other actors); preserves properties
(3) `+` - equivalent to `Assembly`
'''

import vedo
import numpy as np

# Define vertices and faces
verts = np.array([(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)])
faces = np.array([(0, 1, 2), (2, 1, 3), (1, 0, 3), (0, 2, 3)])
# Create a tetrahedron and a copy
mesh = vedo.Mesh([verts, faces], c='red')
mesh2 = mesh.clone().x(15).y(15).c('blue')  # Create a copy, shift it; change color

# Merge: creates a new mesh, color of the second mesh is lost
mesh_all = vedo.merge(mesh, mesh2)
print('1. Type:', type(mesh_all))
# Show
plotter = vedo.show(mesh_all, viewup='z', axes=1)  # -> all red
plotter.close()

# Assembly: groups meshes
mesh_all = vedo.assembly.Assembly(mesh, mesh2)
print('2. Type:', type(mesh_all))
# Show
plotter = vedo.show(mesh_all, viewup='z', axes=1)  # -> red and blue
plotter.close()

# Equivalently, "+" also creates an Assembly
mesh_all = mesh + mesh2
print('3. Type:', type(mesh_all))
# Show
plotter = vedo.show(mesh_all, viewup='z', axes=1)  # -> red and blue
plotter.close()
