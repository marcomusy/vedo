"""Apply a vector-valued point load
to a corner of a linear-elastic cube.
"""
# Credit https://fenicsproject.discourse.group/t/
#applying-pointsource-at-two-different-vectors/1459/2
from dolfin import *
from vedo.dolfin import plot, Text2D

BULK_MOD = 1.0
SHEAR_MOD = 1.0

mesh = UnitCubeMesh(10, 10, 10)
VE = VectorElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, VE)

# Constrain normal displacement on two sides:
def boundary1(x, on_boundary):
    return on_boundary and near(x[1], 0.0)
bc1 = DirichletBC(V.sub(1), Constant(0.0), boundary1)

def boundary2(x, on_boundary):
    return on_boundary and near(x[0], 0.0)
bc2 = DirichletBC(V.sub(0), Constant(0.0), boundary2)

# Solve linear elasticity with point load at upper-right corner:
u = TrialFunction(V)
v = TestFunction(V)

eps = 0.5 * (grad(u) + grad(u).T)
I = Identity(3)
sigma = BULK_MOD*tr(eps)*I + 2*SHEAR_MOD*(eps-tr(eps)*I/3)

a = inner(sigma, grad(v)) * dx
L = inner(Constant((0,0,0)), v) * dx

# Assemble:
A = assemble(a)
B = assemble(L)

# Apply point sources:
ptSrcLocation = Point(1-DOLFIN_EPS, 1-DOLFIN_EPS)

# Vectorial point load:
f = [0.01, 0.02]

# Distinct point sources for x- and y-components
ptSrc_x = PointSource(V.sub(0), ptSrcLocation, f[0])
ptSrc_y = PointSource(V.sub(1), ptSrcLocation, f[1])
ptSrcs = [ptSrc_x, ptSrc_y]

# Apply to RHS of linear system:
for ptSrc in ptSrcs:
    ptSrc.apply(B)

# Apply BCs:
for bc in [bc1, bc2]:
    bc.apply(A)
    bc.apply(B)

# Solve:
u = Function(V)
solve(A, u.vector(), B)

plot(u, mode='displacement')