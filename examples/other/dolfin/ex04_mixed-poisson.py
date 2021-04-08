"""Solving Poisson equation using
a mixed (two-field) formulation."""
# https://fenicsproject.org/docs/dolfin/2018.1.0/python/demos/mixed-poisson
from dolfin import *

# Create mesh
mesh = UnitSquareMesh(30, 30)

# Define finite elements spaces and build mixed space
BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
W   = FunctionSpace(mesh, BDM * DG)

# Define trial and test functions
(sigma, u) = TrialFunctions(W)
(tau, v)   = TestFunctions(W)

# Define source function
f = Expression("10*exp(-(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2))/0.02)", degree=2)

# Define variational form
a = (dot(sigma, tau) + div(tau) * u + div(sigma) * v) * dx
L = -f * v * dx

# Define function G such that G \cdot n = g
class BoundarySource(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = sin(5 * x[0])
        values[0] = g * n[0]
        values[1] = g * n[1]
    def value_shape(self):
        return (2,)
G = BoundarySource(mesh, degree=2)

# Define essential boundary
def boundary(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS
bc = DirichletBC(W.sub(0), G, boundary)

# Compute solution
w = Function(W)
solve(a == L, w, bc)
(sigma, u) = w.split()


########################################################### vedo
from vedo.dolfin import plot, Text3D

# Plot solution on mesh, and warp z-axis by the scalar value
plot(u, warpZfactor=0.8, legend='u', text=__doc__)

# # Plot the sigma vector on the mesh. Try also mode='arrows'
# msg = Text3D("> plot(sigma, mode='mesh lines', warpZfactor= -0.2)", c='w')
# plot(sigma, msg,
#      mode='mesh lines',
#      warpZfactor=-0.2,    # rise mesh in z based on scalar value
#      scale=0.03,          # scale the lines or arrows
#      new=True,            # new window
#     )

