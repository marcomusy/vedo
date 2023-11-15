"""Cut an UnstructuredGrid with a Mesh"""
from vedo import *

ug1 = UnstructuredGrid(dataurl+'ugrid.vtk')
print(ug1)

cyl = Cylinder(r=3, height=7).x(3).wireframe()
ug2 = ug1.clone().cut_with_mesh(cyl)

ug1.wireframe()
show(ug1, ug2, cyl, __doc__, axes=1).close()
