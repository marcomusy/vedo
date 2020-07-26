'''
compute_collision() will compute the collision of all the entities with
a Point while compute_first_collision() will always return its first entry.
Especially if a point is on an element edge this can be tricky.
You may also want to compare with the Cell.contains(Point) tool.
'''
# Script by Rudy at https://fenicsproject.discourse.group/t/
#           any-function-to-determine-if-the-point-is-in-the-mesh/275/3
import dolfin
from vedo.dolfin import plot
from vedo import printc, pointcloud

n  = 4
Px = 0.5
Py = 0.5
mesh = dolfin.UnitSquareMesh(n, n)
bbt = mesh.bounding_box_tree()
collisions = bbt.compute_collisions(dolfin.Point(Px, Py))
collisions1st = bbt.compute_first_entity_collision(dolfin.Point(Px, Py))
printc("collisions    : ", collisions)
printc("collisions 1st: ", collisions1st)

for cell in dolfin.cells(mesh):
    contains = cell.contains(dolfin.Point(Px, Py))
    printc("Cell", cell.index(), "contains P:", contains, c=contains)

###########################################
pt = pointcloud.Point([Px, Py], c='blue')

plot(mesh, pt, text=__doc__)
