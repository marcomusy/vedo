"""Poisson equation with Dirichlet conditions.

  -Laplace(u) = f    in the unit square
            u = uD   on the boundary

  uD = 1 + x^2 + 2*y^2
  (f = -6)
"""
########################################################### fenics
import numpy as np
from fenics import *

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "P", 1)

# Define boundary condition
uD = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)
bc = DirichletBC(V, uD, "on_boundary")

# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
u = Function(V)
f = Constant(-6.0)

# Compute solution
solve( dot(grad(w), grad(v))*dx == f*v*dx,  u, bc)

f = r'-\nabla^{2} u=f'

########################################################### vedo
from vedo.dolfin import plot, clear, histogram
from vedo import Latex

l = Latex(f, s=0.2, c='w').addPos(.6,.6,.1)

plot(u, l, cmap='jet', scalarbar='h', text=__doc__)

# Now show uD values on the boundary of a much finer mesh
clear()
bmesh = BoundaryMesh(UnitSquareMesh(80, 80), "exterior")
plot(uD, bmesh, cmap='cool', ps=5, legend='boundary') # ps = point size


# now make some nonsense plot with the same plot() function
yvals = u.compute_vertex_values(mesh)
xvals = np.arange(len(yvals))
plt = plot(xvals, yvals, 'go-')
plt.show(new=True)

# and a histogram
hst = histogram(yvals)
hst.show(new=True)
