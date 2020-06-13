from dolfin import *

mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, "triangle", 2, 2)
editor.init_vertices(3)
editor.add_vertex(0, [-1, 0])
editor.add_vertex(1, [ 1, 0])
editor.add_vertex(2, [ 0, 1])
editor.init_cells(1)
editor.add_cell(0, [0, 1, 2])
editor.close()
mesh.init()

W = FunctionSpace(mesh, "BDM", 1)

v = Expression(("0", "x[0]*x[0]"), degree=2)

vi = interpolate(v, W)

from vedo.dolfin import plot
plot(vi, scalarbar="horizontal", style="meshlab")
