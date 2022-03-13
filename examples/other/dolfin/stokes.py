"""
This demo solves the Stokes equations, using quadratic elements for
the velocity and first degree elements for the pressure (Taylor-Hood elements).
"""
# Credits:
# https://github.com/pf4d/fenics_scripts/blob/master/cbc_block/stokes.py
from dolfin import *
import numpy as np
from vedo.dolfin import plot, dataurl, download
from vedo import Latex

# Load mesh and subdomains
fpath = download(dataurl+"dolfin_fine.xml")
mesh = Mesh(fpath)

fpath = download(dataurl+"dolfin_fine_subdomains.xml.gz")
sub_domains = MeshFunction("size_t", mesh, fpath)

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

solve(a == L, w, bcs)

# Split the mixed solution using a shallow copy
(u, p) = w.split()

##################################################################### vedo
f = r'-\nabla \cdot(\nabla u+p I)=f ~\mathrm{in}~\Omega'
formula = Latex(f, pos=(0.55,0.45,-.05), s=0.1)

plot(u, formula, at=0, N=2,
     mode='mesh and arrows', scale=.03,
     wireframe=True, scalarbar=False, style=1)
plot(p, at=1, text="pressure", cmap='rainbow', interactive=False)


##################################################################### streamlines
# A list of seed points (can be automatic: just comment out 'probes')
ally = np.linspace(0,1, num=100)
probes = np.c_[np.ones_like(ally), ally, np.zeros_like(ally)]

plot(u,
     mode='mesh with streamlines',
     streamlines={'tol':0.02,            # control density of streams
                  'lw':2,                # line width
                  'direction':'forward', # direction of integration
                  'maxPropagation':1.2,  # max length of propagation
                  'probes':probes,       # custom list of point in space as seeds
                 },
     c='white',                          # mesh color
     alpha=0.3,                          # mesh alpha
     lw=0,                               # mesh line width
     wireframe=True,                     # show as wireframe
     bg='blackboard',                    # background color
     new=True,                           # new window
     pos=(200,200),                      # window position on screen
     )
