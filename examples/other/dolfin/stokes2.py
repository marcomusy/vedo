from dolfin import *
from mshr import *

b = 0.7
embryo = Ellipse(Point(0.0, 0.0), 1, b)
mesh = generate_mesh(embryo, 32)

# Define function spaces
P2 = VectorElement("CG", triangle, 2)
P1 = FiniteElement("CG", triangle, 1)
TH = MixedElement([P2, P1])
W  = FunctionSpace(mesh, TH)
g  = Constant(0.0)
mu = Constant(1.0)
force = Constant((0.0, 0.0))

# Specify Boundary Conditions
flow_profile = ("-sin(atan2(x[1]/b, x[0]))*sin(nharmonic*atan2(x[1]/b, x[0]))",
                "+cos(atan2(x[1]/b, x[0]))*sin(nharmonic*atan2(x[1]/b, x[0]))")
bc = DirichletBC(W.sub(0),
                 Expression(flow_profile, degree=2, b=b, nharmonic=2),
                 "on_boundary")

# Define trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

def epsilon(u):
    return grad(u) + nabla_grad(u)
a = inner(mu*epsilon(u) + p*Identity(2), epsilon(v))*dx -div(u)*q*dx -1e-10*p*q*dx
L = dot(force, v)*dx + g*q*dx

# Solve system
U = Function(W)
solve(a == L, U, bc)

# Get sub-functions
u, p = U.split()

from vedo.dolfin import plot
plot(u,
     mode='mesh and arrows',
     scale=0.1,
     warpZfactor=-0.1,
     lw=0,
     scalarbar='horizontal',
     axes={'xLabelSize':0.01,'yLabelSize':0.01, 'ztitle':''},
     title="Velocity")
