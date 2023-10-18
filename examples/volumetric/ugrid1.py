"""Cut an UnstructuredGrid with a mesh"""
from vedo import *

ug1 = UGrid(dataurl+'ugrid.vtk')
ms1 = ug1.clone().tomesh().wireframe()

cyl = Cylinder(r=3, height=7).x(3).wireframe()
ms2 = ug1.cut_with_mesh(cyl).tomesh().cmap('jet')

show(ms1, ms2, cyl, __doc__, axes=1).close()
