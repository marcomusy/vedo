'''
Mark mesh with boundary function
'''
from dolfin import *

mesh = UnitCubeMesh(5,5,5)
V = FunctionSpace(mesh, "Lagrange", 1)

class left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS
left = left()

tcond = MeshFunction("size_t", mesh, 0)
tcond.set_all(0)
left.mark(tcond, 1)

##################################
from vedo.dolfin import plot
plot(tcond, cmap='cool', elevation=20, text=__doc__)
