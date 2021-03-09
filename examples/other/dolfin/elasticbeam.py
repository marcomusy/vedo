"""A beam deforming under its own weight."""

from dolfin import *

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
from vedo.dolfin import *

plot(u, mode="displaced mesh",
     text=__doc__,
     scalarbar=False,
     axes=1,
     viewup='z')
#exportWindow('elasticbeam1.x3d') # generate a html test page

txt = Text3D("Von Mises stress intensity", pos=(0.1,.12,0), s=0.03, c='white')
plot(von_Mises, txt, cmap='plasma', scalarbar=False, new=True)
#exportWindow('elasticbeam2.x3d') # generate a html test page

txt = Text3D("Magnitude of displacement", pos=(0.1,.12,0), s=0.03, c='white')
plot(u_magnitude, txt, scalarbar=False, new=True)
#exportWindow('elasticbeam3.x3d') # generate a html test page

