"""Compute some quantity in each node
of a mesh (by looping on the nodes)
and then build a piecewise linear
function with computed nodal values."""
from dolfin import *
from vedo.dolfin import plot


def f(coordinate):
    return coordinate[0] * coordinate[1]

mesh = UnitSquareMesh(10, 10)

V = FunctionSpace(mesh, "CG", 1)
g = Function(V)

coords = V.tabulate_dof_coordinates()

for i in range(V.dim()):
    g.vector()[i] = f(coords[i])

plot(g, text=__doc__)
