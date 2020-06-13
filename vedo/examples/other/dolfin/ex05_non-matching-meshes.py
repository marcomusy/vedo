"""
Interpolate functions between
finite element spaces on non-matching meshes.
"""
# https://fenicsproject.org/docs/dolfin/2018.1.0/python/demos
# /nonmatching-interpolation/demo_nonmatching-interpolation.py.html
from dolfin import *

mesh1 = UnitSquareMesh(16, 16)
mesh3 = UnitSquareMesh(64, 64)

P1 = FunctionSpace(mesh1, "Lagrange", 1)
P3 = FunctionSpace(mesh3, "Lagrange", 3)

v = Expression("sin(10*x[0])*sin(10*x[1])", degree=5)

# Create function on P3 and interpolate v
v3 = Function(P3)
v3.interpolate(v)

# Create function on P1 and interpolate v3
v1 = Function(P1)
v1.interpolate(v3)


######################################### vedo
from vedo.dolfin import plot

# Plot v1 and v3 on 2 synced renderers
plot(v1, at=0, N=2, text='coarse mesh')
plot(v3, at=1, text='finer  mesh', interactive=True)
