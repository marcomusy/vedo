from dolfin import *

print('Test elastic_beam.py')

# Scaled variables
l, w = 1, 0.1
mu_, lambda_ = 1, 1
rho = 10
gamma = (w/l)**2
wind = (0, 0.0, 0)

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(l, w, w), 50, 5, 5)
V = VectorFunctionSpace(mesh, "P", 1)

# Define boundary condition
def clamped_boundary(x, on_boundary):
    return on_boundary and (near(x[0], 0) or near(x[0], l))
bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# Define strain and stress
def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lambda_ * nabla_grad(u) * Identity(3) + 2 * mu_ * epsilon(u)


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0, 0, -rho * gamma))
T = Constant(wind)
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(T, v) * ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

s = sigma(u) - (1.0 / 3) * tr(sigma(u)) * Identity(3)  # deviatoric stress
von_Mises = sqrt(3.0 / 2 * inner(s, s))
V = FunctionSpace(mesh, "P", 1)
von_Mises = project(von_Mises, V)
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)

################################ Plot solution
from vtkplotter.dolfin import *

plot(u, mode="displaced mesh",
     text=__doc__,
     scalarbar=False,
     axes=1,
     bg='white',
     viewup='z',
     offscreen=1)

#################################################################################
import numpy as np
from vtkplotter import settings, screenshot
actor = settings.plotter_instance.actors[0]
solution = actor.scalars(0)

screenshot('elasticbeam.png')

print('ArrayNames', actor.getArrayNames())
print('min', 'mean', 'max, N:')
print(np.min(solution), np.mean(solution), np.max(solution), len(solution))

assert len(solution) == 1836
assert np.isclose(np.min(solution) , 0., rtol=1e-05)
assert np.isclose(np.mean(solution), 0.054582344380721216, rtol=1e-05)
assert np.isclose(np.max(solution) , 0.10077310100074487,  rtol=1e-05)

coords = actor.coordinates()
print('Coordinates check')
print('min', 'mean', 'max, N:')
print(np.min(coords), np.mean(coords), np.max(coords), len(coords))
assert len(solution) == 1836
assert np.isclose(np.min(coords) , -0.10071969054703424, rtol=1e-05)
assert np.isclose(np.mean(coords), 0.18263017040948404, rtol=1e-05)
assert np.isclose(np.max(coords) , 1.0,  rtol=1e-05)


