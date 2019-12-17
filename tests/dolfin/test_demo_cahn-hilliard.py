import numpy as np
from vtkplotter import settings

import random
random.seed(0)

from dolfin import *
from vtkplotter.dolfin import plot, screenshot


print('Test demo_cahn-hilliard')

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        pass
        #super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.63 + 0.02 * (0.5 - random.random())
        values[1] = 0.0
    def value_shape(self): return (2,)

# Class for interfacing with the Newton solver
class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x): assemble(self.L, tensor=b)
    def J(self, A, x): assemble(self.a, tensor=A)

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt    = 5.0e-06  # time step
# time stepping family,
# e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
theta = 0.5

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and define function spaces
mesh = UnitSquareMesh(60, 60)
# mesh = UnitSquareMesh.create(60, 60, CellType.Type.triangle)
# V = FunctionSpace(mesh, "Lagrange", 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1 * P1)

# Define trial and test functions
du   = TrialFunction(ME)
q, v = TestFunctions(ME)

# Define functions
u  = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

# Split mixed functions
dc, dmu = split(du)
c, mu   = split(u)
c0, mu0 = split(u0)

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)

# Compute the chemical potential df/dc
c = variable(c)
f = 100 * c ** 2 * (1 - c) ** 2
mu_mid = (1 - theta) * mu0 + theta * mu

# Weak statement of the equations
L0 = c * q - c0 * q + dt * dot(grad(mu_mid), grad(q))
L1 = mu * v - diff(f, c) * v - lmbda * dot(grad(c), grad(v))
L  = (L0 + L1) * dx

# Compute directional derivative about u in the direction of du
a = derivative(L, u, du)

# Create nonlinear problem and Newton solver
problem = CahnHilliardEquation(a, L)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

# Step in time   
t = 0
T = 10*dt
scalarbar = False
while t < T:
    t += dt
    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector())
    if t==T:
        scalarbar = 'horizontal'

    plot(u.split()[0],
         z=t*2e4,
         add=True, # do not clear the canvas
         style=0,
         lw=0,
         scalarbar=scalarbar,
         elevation=-3, # move camera a bit
         azimuth=1,
         text='time: '+str(t*2e4),
         lighting='plastic',
         offscreen=1)
    break
    

#################################################################################
actor = settings.plotter_instance.actors[0]
solution = actor.scalars(0)

screenshot('demo_cahn-hilliard.png')

print('ArrayNames', actor.getArrayNames())
print('min', 'mean', 'max, N:')
print(np.min(solution), np.mean(solution), np.max(solution), len(solution))

assert len(solution) == 3721
assert np.isclose(np.min(solution) , 0.6197379993778627, rtol=1e-05)
assert np.isclose(np.mean(solution), 0.6300829130568726, rtol=1e-05)
assert np.isclose(np.max(solution) , 0.6412288864268026, rtol=1e-05)

