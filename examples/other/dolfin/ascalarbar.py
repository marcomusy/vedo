"""#Control scalar bar range.
> plot(u, mode='color', vmin=-3, vmax=3, style=1)

Available styles:
0. vtk
1. matplotlib
2. meshlab
3. paraview
4. bw
"""
from dolfin import *

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, 'Lagrange', 1)
f = Expression('10*(x[0]+x[1]-1)', degree=1)
u = interpolate(f, V)


################################## vedo
from vedo.dolfin import plot

plot(u, mode='color', vmin=-3, vmax=3, style=1, text=__doc__)

