
from vedo import *

ug1 = UGrid(dataurl+'ugrid.vtk')

ug2= ug1.clone().tomesh().wireframe()

cyl = Cylinder(r=3, height=7).x(5).wireframe()
ug1.cutWithMesh(cyl)

show(ug1, ug2, cyl, axes=1).close()
