"""
Poisson equation with Dirichlet conditions.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

  u_D = 1 + x^2 + 2y^2
  (f = -6)
"""
########################################################### fenics
from fenics import *

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "P", 1)

# Define boundary condition
u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)
bc = DirichletBC(V, u_D, "on_boundary")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)


########################################################### vtkplotter
from vtkplotter.dolfin import *

msg = Text(__doc__, c="white")

# u=u triggers coloring with default color map
pts = MeshActor(u, wire=False)
pts.lineWidth(1).addScalarBar()
show(pts, msg)

##### Now show u_D values on the boundary of a much finer mesh
bmesh = BoundaryMesh(UnitSquareMesh(80, 80), "exterior")

bpts = MeshPoints(u_D, bmesh)
show(bpts)
