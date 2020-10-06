'''
Combine mesh objects with
(1) `mesh.merge`
(2) `+`
(3) `assembly.Assembly`
'''

import vedo
import numpy as np

# Create a tetrahydron
verts = np.array([(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)])
faces = np.array([(0, 1, 2), (2, 1, 3), (1, 0, 3), (0, 2, 3)])

# Build the polygonal Mesh object:
mesh = vedo.Mesh([verts, faces], c='red')
# Create a copy, shift it; change color
mesh2 = mesh.clone().x(15).y(15).c('blue')
# mesh2.backColor('violet').lineColor('tomato').lineWidth(2)

# Merge creates a new mesh, color - red
mesh_all = vedo.merge(mesh, mesh2)
print('1. Type:', type(mesh_all))
# Show
plotter = vedo.show(mesh_all, viewup='z', axes=1)  # -> all red
plotter.close()

# "+" creates an Assembly
mesh_all = mesh + mesh2
print('2. Type:', type(mesh_all))
# Show
plotter = vedo.show(mesh_all, viewup='z', axes=1)  # -> red and blue
plotter.close()

# Equivalently, can call Assembly
mesh_all = vedo.assembly.Assembly(mesh, mesh2)
print('3. Type:', type(mesh_all))
# Show
plotter = vedo.show(mesh_all, viewup='z', axes=1)  # -> red and blue
plotter.close()
