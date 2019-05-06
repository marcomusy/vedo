"""
A clamped beam deforming
under its own weight.
"""
from dolfin import *

# Scaled variables
L = 1
W = 0.1
mu = 1
rho = 1
delta = W / L
gamma = 1.2 * delta ** 2
beta = 1.25
lambda_ = beta
g = gamma

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 15, 3, 3)
V = VectorFunctionSpace(mesh, "P", 1)

# Define boundary condition
tol = 1e-14
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol
bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# Define strain and stress
def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lambda_ * nabla_grad(u) * Identity(d) + 2 * mu * epsilon(u)


# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)
f = Constant((0, 0, -rho * g))
T = Constant((0, 0, 0))
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(T, v) * ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

s = sigma(u) - (1.0 / 3) * tr(sigma(u)) * Identity(d)  # deviatoric stress
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
     viewup='z')
exportWindow('elasticbeam1.x3d') # generate a html test page

txt = Text("Von Mises stress intensity", pos=(0.1,.12,0), s=0.03, c='white')
plot(von_Mises, txt, cmap='plasma', scalarbar=False, newPlotter=True)
exportWindow('elasticbeam2.x3d')

txt = Text("Magnitude of displacement", pos=(0.1,.12,0), s=0.03, c='white')
plot(u_magnitude, txt, scalarbar=False, newPlotter=True)
exportWindow('elasticbeam3.x3d')
