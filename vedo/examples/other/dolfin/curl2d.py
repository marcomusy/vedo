#Magnetostatic equation that solves for magnetic vector potential,
#where the reluctivity matrix can be defined as inverse of permeability matrix..
#https://fenicsproject.discourse.group/t/anisotropic-material-definition-and-results-issue/1051
from dolfin import *
from mshr import *
from scipy import constants
from vedo.dolfin import plot

domain = Rectangle(Point(-10, -10), Point(10, 10))
mesh = generate_mesh(domain, 64)

# function space
V = FunctionSpace(mesh, "P", 1)

# boundary conditions
walls = "on_boundary && (near(abs(x[0]), 10.0) || near(abs(x[1]), 10.0))"
bc = DirichletBC(V, Constant(0.0), walls)

tol = 1e-6

# Wire
class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] ** 2 + x[1] ** 2 <= 4 - tol

# Space
class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] ** 2 + x[1] ** 2 > 4 + tol

def curl2D(v):
    return as_vector((v.dx(1), -v.dx(0)))


materials = MeshFunction("size_t", mesh, mesh.topology().dim())

subdomain_0 = Omega_0()
subdomain_1 = Omega_1()
subdomain_0.mark(materials, 0)
subdomain_1.mark(materials, 1)

dx = Measure("dx", domain=mesh, subdomain_data=materials)

A_z = Function(V)  # magnetic vector potential
v = TestFunction(V)

J = 5.0e6
# anisotropic material parameters, reluctivity = 1/constants.mu_0
reluctivity = as_matrix(
    ((1 / (constants.mu_0 * 1000), 0),
     (0, 1 / (constants.mu_0 * 1)))
)

F = inner(reluctivity * curl2D(A_z), curl2D(v)) * dx - J * v * dx(0)
solve(F == 0, A_z, bc)

W = VectorFunctionSpace(mesh, "P", 1)

Bx =  A_z.dx(1)
By = -A_z.dx(0)
B = project(as_vector((Bx, By)), W)

plot(B, mode='mesh and arrows',
     style=2,
     scale=0.01,
     lw=0,
     warpZfactor=-0.01,
     )
