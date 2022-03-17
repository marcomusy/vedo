"""Find cells within specified bounds in x, y and/or z."""
from vedo import *

mesh = Mesh(dataurl+'shark.ply').normalize()
mesh.color('aqua').lineWidth(1)

z1, z2 = -1.5, -1.0

ids = mesh.findCellsWithin(zbounds=(z1,z2))
printc('IDs of cells within bounds:\n', ids, c='g')

p1 = Plane(normal=(0,0,1), s=[2,2]).z(z1)
p2 = p1.clone().z(z2)

show(mesh, p1, p2, __doc__, axes=9).close()
