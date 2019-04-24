"""
This demo solves the Stokes equations, using quadratic elements for
the velocity and first degree elements for the pressure (Taylor-Hood elements).
"""
# Credits:
# https://github.com/pf4d/fenics_scripts/blob/master/cbc_block/stokes.py
from dolfin import *
from time import time
from vtkplotter.dolfin import plot, datadir, Latex

t0 = time()
print("calculating... please wait...")

# Load mesh and subdomains
mesh = Mesh(datadir+"dolfin_fine.xml")
sub_domains = MeshFunction("size_t", mesh,
                           datadir+"dolfin_fine_subdomains.xml.gz")

# Define function spaces
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)
bcs = [bc0, bc1]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = inner(f, v)*dx
w = Function(W)

solve(a == L, w, bcs, solver_parameters={'linear_solver' : 'mumps'})
tf = time()
print("time to solve:", tf-t0)

# Split the mixed solution using a shallow copy
(u, p) = w.split()

######################################################## vtkplotter:
f = r'-\nabla \cdot(\nabla u+p I)=f ~\mathrm{in}~\Omega'
formula = Latex(f, pos=(0.55,0.45,-.05), s=0.1)

plot(u, formula, at=0, N=2, text="velocity", mode='mesh and arrows',
	 scale=.03, wire=1, scalarbar=False, style=1)
plot(p, at=1, text="pressure", cmap='jet')
