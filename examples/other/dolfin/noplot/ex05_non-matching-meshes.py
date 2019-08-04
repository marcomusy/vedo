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


######################################### vtkplotter
from vtkplotter.dolfin import *

s1 = MeshActor(v1).lineWidth(0.5).wireframe(False)
s3 = MeshActor(v3).lineWidth(0.5).wireframe(False)

show(s1, s3, N=2) # distribute s1 and s3 on 2 synced renderers
