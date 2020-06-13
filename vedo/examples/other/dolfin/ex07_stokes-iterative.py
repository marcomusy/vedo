"""
Stokes equations with an iterative solver.
"""
# https://fenicsproject.org/docs/dolfin/2018.1.0/python/demos/
#  stokes-iterative/demo_stokes-iterative.py.html
from dolfin import *


mesh = UnitCubeMesh(10, 10, 10)

# Build function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# Boundaries
def right(x, on_boundary):
    return x[0] > (1.0 - DOLFIN_EPS)

def left(x, on_boundary):
    return x[0] < DOLFIN_EPS

def top_bottom(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"), degree=2)
bc1 = DirichletBC(W.sub(0), inflow, right)

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0.0, 0.0, 0.0))
a = inner(grad(u), grad(v)) * dx + div(v) * p * dx + q * div(u) * dx
L = inner(f, v) * dx

# Form for use in constructing preconditioner matrix
b = inner(grad(u), grad(v)) * dx + p * q * dx

# Assemble system
A, bb = assemble_system(a, L, [bc0, bc1])

# Assemble preconditioner system
P, btmp = assemble_system(b, L, [bc0, bc1])

# Create Krylov solver and AMG preconditioner
if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
solver = KrylovSolver(krylov_method, "amg")

# Associate operator (A) and preconditioner matrix (P)
solver.set_operators(A, P)

# Solve
U = Function(W)
solver.solve(U.vector(), bb)

# Get sub-functions
u, p = U.split()
pressures = p.compute_vertex_values(mesh)


#################################################### vedo
from vedo.dolfin import plot, printHistogram

# Plot u and p solutions on N=2 synced renderers
plot(u, mode='mesh arrows', at=0, N=2, legend='velocity',
     scale=0.1, wireframe=1, lw=0.03, alpha=0.5, scalarbar=False)

printHistogram(pressures, title='pressure histo', logscale=True, c=1)

plot(p, mode='mesh', at=1, N=2, legend='pressure', interactive=True)
