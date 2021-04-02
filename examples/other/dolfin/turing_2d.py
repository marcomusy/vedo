#
# https://fenicsproject.org/qa/8612/difficulties-with-solving-the-gray-scott-model
from dolfin import *
import numpy as np

# Set parameters
D_u = 8.0e-05
D_v = 4.0e-05
c = 0.022
k = 0.055

dt = 12.0
t_max = 100000

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
set_log_level(30)


# Class representing the initial conditions
class InitialConditions(UserExpression):
    def eval(self, val, x):
        if between(x[0], (1.0, 1.5)) and between(x[1], (1.0, 1.5)):
            val[1] = 0.25*np.power(np.sin(4*np.pi*x[0]), 2)*np.power(np.sin(4*np.pi*x[1]), 2)
            val[0] = 1 - 2*val[1]
        else:
            val[1] = 0
            val[0] = 1
    def value_shape(self):
        return (2,)

# Class for interfacing with the Newton solver
class GrayScottEquations(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

# Define mesh and function space
p0 = Point(0.0, 0.0)
p1 = Point(2.5, 2.5)
mesh = RectangleMesh(p0, p1, 64, 64)
V = VectorFunctionSpace(mesh, 'CG', 2)

# Define functions
W_init = InitialConditions(degree = 1)
phi    = TestFunction(V)
dp     = TrialFunction(V)
W0     = Function(V)
W      = Function(V)
# Interpolate initial conditions and split functions
W0.interpolate(W_init)
q, p   = split(phi)
u,  v  = split(W)
u0, v0 = split(W0)

# Weak statement of the equations
F1 = u*q*dx -u0*q*dx +D_u*inner(grad(u), grad(q))*dt*dx +u*v*v*q*dt*dx -c*(1-u)*q*dt*dx
F2 = v*p*dx -v0*p*dx +D_v*inner(grad(v), grad(p))*dt*dx -u*v*v*p*dt*dx +(c+k)*v*p*dt*dx
F = F1 + F2

# Compute directional derivative about W in the direction of dp (Jacobian)
a = derivative(F, W, dp)

# Create nonlinear problem and Newton solver
problem = GrayScottEquations(a, F)
solver = NewtonSolver()
#solver.parameters["linear_solver"] = "lu"
#solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-3

from vedo.dolfin import *

t = 0.0
W.assign(W0)
while (t < t_max):
    t += dt
    solver.solve(problem, W.vector())
    W0.assign(W)
    u_out, v_out = W.split()
    plot(u_out, text="time = "+str(t),
         lw=0, warpZfactor=-0.1,
         vmin=0, vmax=1, scalarbar=False, interactive=False)

interactive()

