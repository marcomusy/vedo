"""
Solution of a particular nonlinear time-dependent
fourth-order equation, known as the Cahn-Hilliard equation.
"""
#https://fenicsproject.org/olddocs/dolfin/1.4.0/python/demo/documented
import random
from dolfin import *
set_log_level(30)


# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
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
from vedo.dolfin import plot

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
         interactive=0 )

plot()
