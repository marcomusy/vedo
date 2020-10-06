'''
Mesh objects can be combined with
(1) `mesh.merge` - creates a new mesh object; this new mesh inherits properties (color, etc.) of the first mesh.
(2) `assembly.Assembly` - groups meshes (or other actors); preserves properties
(3) `+` - equivalent to `Assembly`
'''
# credits: https://github.com/icemtel
import vedo

# Define vertices and faces
verts = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)]
faces = [(0, 1, 2), (2, 1, 3), (1, 0, 3), (0, 2, 3)]
# Create a tetrahedron and a copy
mesh1 = vedo.Mesh([verts, faces], c='red')
mesh2 = mesh1.clone().pos(15,15,0).c('blue')  # Create a copy, position it; change color

# Merge: creates a new mesh, fusion of the 2 inputs. Color of the second mesh is lost.
mesh_all = vedo.merge(mesh1, mesh2)
print('1. Type:', type(mesh_all))
plotter = vedo.show("mesh.merge(mesh1, mesh2) creates a single new Mesh object",
                    mesh_all, viewup='z', axes=1)  # -> all red
plotter.close()

# Assembly: groups meshes. Objects keep their individuality (can be later unpacked).
mesh_all = vedo.Assembly(mesh1, mesh2)
print('2. Type:', type(mesh_all))
plotter = vedo.show("Assembly(mesh1, mesh2) groups meshes preserving their properties",
                    mesh_all, viewup='z', axes=1)  # -> red and blue
plotter.close()

# Equivalently, "+" also creates an Assembly
mesh_all = mesh1 + mesh2
print('3. Type:', type(mesh_all))
plotter = vedo.show("mesh1+mesh2 operator is equivalent to Assembly()",
                    mesh_all, viewup='z', axes=1)  # -> red and blue
plotter.close()
