"""
FEniCS tutorial demo program: Deflection of a membrane.

  -Laplace(w) = p  in the unit circle
            w = 0  on the boundary

The load p is a Gaussian function centered at (0, 0.6).
"""
from fenics import *
from mshr import Circle, generate_mesh

# Create mesh and define function space
domain = Circle(Point(0, 0), 1)
mesh = generate_mesh(domain, 64)
V = FunctionSpace(mesh, 'P', 2)
w_D = Constant(0)

def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, w_D, boundary)

# Define load
p = Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))',
               degree=1, beta=8, R0=0.6)

# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(w), grad(v))*dx
L = p*v*dx

# Compute solution
w = Function(V)
solve(a == L, w, bc)

p = interpolate(p, V)

# Curve plot along x = 0 comparing p and w
import numpy as np
tol = 0.001  # avoid hitting points outside the domain
y = np.linspace(-1 + tol, 1 - tol, 101)
points = [(0, y_) for y_ in y]  # 2D points
w_line = np.array([w(point) for point in points])
p_line = np.array([p(point) for point in points])


#######################################################################
from vedo.dolfin import plot
from vedo import Line, Latex

pde = r'-T \nabla^{2} D=p, ~\Omega=\left\{(x, y) | x^{2}+y^{2} \leq R\right\}'
tex = Latex(pde, pos=(0,1.1,.1), s=0.2, c='w')

wline = Line(y, w_line*10, c='white', lw=4)
pline = Line(y, p_line/ 4, c='lightgreen', lw=4)

plot(w, wline, tex, at=0, N=2, bg='bb', text='Deflection')
plot(p, pline, at=1, bg='bb', text='Load')

