#
# https://fenicsproject.org/qa/8612/difficulties-with-solving-the-gray-scott-model
from dolfin import *
import numpy as np
import mshr

# Set parameters
D_u = 8.0e-05
D_v = 4.0e-05
c = 0.024
k = 0.06

dt = 1.0
t_max = 100

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
set_log_level(30)

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def eval(self, val, x):
        val[0] = np.random.rand()
        val[1] = np.random.rand()
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
p1 = Point(1.0, 1.0)
p0 = Point(0.0, 0.0, 0.0)
sphere = mshr.Sphere(p0, 1.0)
mesh = mshr.generate_mesh(sphere, 16)

V_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([V_ele, V_ele]))

# Define functions
W_init = InitialConditions(degree=1)
phi = TestFunction(V)
dp = TrialFunction(V)
W0 = Function(V)
W = Function(V)

# Interpolate initial conditions and split functions
W0.interpolate(W_init)
q, p = split(phi)
u, v = split(W)
u0, v0 = split(W0)

# Weak statement of the equations
F1 = u*q*dx -u0*q*dx +D_u*inner(grad(u),grad(q)) *dt*dx +u*v*v*q*dt*dx -c*(1-u)*q*dt*dx
F2 = v*p*dx -v0*p*dx +D_v*inner(grad(v),grad(p)) *dt*dx -u*v*v*p*dt*dx +(c+k)*v*p*dt*dx
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
while t < t_max:
    t += dt
    solver.solve(problem, W.vector())
    W0.assign(W)
    u_out, v_out = W.split()
    plot(u_out, Text2D("time = "+str(t)),
         vmin=0, vmax=1, scalarbar=False, interactive=False)

interactive()
