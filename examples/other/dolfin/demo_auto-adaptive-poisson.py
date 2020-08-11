# Auto adaptive Poisson equation
# ==============================
#
# In this demo we will use goal oriented adaptivity and error control
# which applies a duality technique to derive error estimates taken
# directly from the computed solution which then are used to weight
# local residuals.

from dolfin import *
set_log_level(30)

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition
u0 = Function(V)
bc = DirichletBC(V, u0, "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1)
g = Expression("sin(5*x[0])", degree=1)
a = inner(grad(u), grad(v))*dx()
L = f*v*dx() + g*v*ds()

# Define function for the solution
u = Function(V)

# Define goal functional (quantity of interest)
M = u*dx()

# Define error tolerance
tol = 1.e-5
problem = LinearVariationalProblem(a, L, u, bc)
solver = AdaptiveLinearVariationalSolver(problem, M)
solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg"
solver.parameters["error_control"]["dual_variational_solver"]["symmetric"] = True
solver.solve(tol)
solver.summary()


from vedo.dolfin import plot

# Plot on 2 synced renderers
plot(u.root_node(), at=0, N=2)
plot(u.leaf_node(), at=1, text='final mesh', interactive=True)
