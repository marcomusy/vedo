from fenics import *
import numpy as np

print('Test poisson' )

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "P", 1)

# Define boundary condition
uD = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)
bc = DirichletBC(V, uD, "on_boundary")

# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
u = Function(V)
f = Constant(-6.0)

# Compute solution
solve( dot(grad(w), grad(v))*dx == f*v*dx,  u, bc)

f = r'-\nabla^{2} u=f'

########################################################### vtkplotter
from vtkplotter.dolfin import plot, Latex, clear, show

l = Latex(f, s=0.2, c='w').addPos(.6,.6,.1)

acts = plot(u, l, cmap='jet', scalarbar='h', returnActorsNoShow=True)

actor = acts[0]

solution = actor.getPointArray(0)

print('ArrayNames', actor.getArrayNames())
print('min', 'mean', 'max:')
print(np.min(solution), np.mean(solution), np.max(solution), len(solution))

assert np.isclose(np.min(solution) , 1.,     atol=1e-03)
assert np.isclose(np.mean(solution), 2.0625, atol=1e-03)
assert np.isclose(np.max(solution) , 4.,     atol=1e-03)
assert len(solution) == 81

print('Test poisson PASSED')
