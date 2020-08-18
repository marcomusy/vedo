"""
Solve the incompressible Navier-Stokes equations
on an L-shaped domain using Chorin's splitting method.
"""
from __future__ import print_function
from dolfin import *
from vedo.dolfin import ProgressBar, plot, download

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False

# Load mesh from file
fpath = download("https://vedo.embl.es/examples/data/lshape.xml.gz")
mesh = Mesh(fpath)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Set parameter values
dt = 0.01
T = 3
nu = 0.01

# Define time-dependent pressure boundary condition
p_in = Expression("sin(3.0*t)", t=0.0, degree=2)

# Define boundary conditions
noslip  = DirichletBC(V, (0, 0), "on_boundary && \
                       (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | \
                       (x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS))")
inflow  = DirichletBC(Q, p_in, "x[1] > 1.0 - DOLFIN_EPS")
outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")
bcu = [noslip]
bcp = [inflow, outflow]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Time-stepping
pb = ProgressBar(0, T, dt, c='green')
for t in pb.range():

    # Update pressure boundary condition
    p_in.t = t

    # Compute tentative velocity step
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "default")

    # Pressure correction
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", prec)

    # Velocity correction
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "bicgstab", "default")

    # Move to next time step
    u0.assign(u1)
    t += dt

    # Plot solution
    plot(u1,
         mode='mesh and arrows',
         text="Velocity of fluid",
         cmap='jet',
         scale=0.3, # unit conversion factor
         scalarbar=False,
         interactive=False)
    pb.print()

plot()


