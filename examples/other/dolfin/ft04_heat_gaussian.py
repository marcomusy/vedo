"""
FEniCS tutorial demo program: Diffusion of a Gaussian hill.

  u'= Laplace(u) + f  in a square domain
  u = u_D             on the boundary
  u = u_0             at t = 0

  u_D = f = 0

The initial condition u_0 is chosen as a Gaussian hill.
"""
# https://fenicsproject.org/pub/tutorial/html/._ftut1006.html
from fenics import *
set_log_level(30)


num_steps = 50  # number of time steps
dt = 0.02       # time step size

# Create mesh and define function space
nx = ny = 30
mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# Define initial value
u_0 = Expression('exp(-5*pow(x[0],2) - 5*pow(x[1],2))', degree=2)
u_n = interpolate(u_0, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

############################################################# vedo
from vedo.dolfin import plot
from vedo import Latex

f = r'\frac{\partial u}{\partial t}=\nabla^2 u+f~\mathrm{in}~\Omega\times(0,T]'
formula = Latex(f, pos=(-.4,-.8, .1), s=0.6, c='w')
formula.crop(0.2, 0.4) # crop top and bottom 20% and 40%

# Time-stepping
u = Function(V)
for n in range(num_steps):

    # Compute solution
    solve(a == L, u, bc)

    # Plot solution
    plot(u, formula, scalarbar=False, interactive=False)

    # Update previous solution
    u_n.assign(u)

plot()
