from __future__ import division, print_function
from dolfin import *
import numpy as np
from vtkplotter import Box, Video, load, Plotter, datadir

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Define mesh
#mesh = Mesh('FEniCS.xml')
mesh = Mesh(datadir+'fenics.xml.gz')


# Sub domain for clamp at left end
def left(x, on_boundary):
    return x[0]<0 and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return x[0]>5.4 and on_boundary

# Elastic parameters
E  = 2000.0
nu = 0.3
mu    = Constant(E / (2.0*(1.0 + nu)))
lmbda = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)))

# Mass density
rho = Constant(1.0)

# Rayleigh damping coefficients
eta_m = Constant(0.)
eta_k = Constant(0.)

# Generalized-alpha method parameters
alpha_m = Constant(0.2)
alpha_f = Constant(0.4)
gamma   = Constant(0.5+alpha_f-alpha_m)
beta    = Constant((gamma+0.5)**2/4.)

# Time-stepping parameters
Nsteps  = 125
T       = .010*Nsteps
dt = Constant(T/Nsteps)

p0 = 2.
cutoff_Tc = T/5
# Define the loading as an expression depending on t
#p = Expression(("0", "0", "t <= tc ? p0*t/tc : 0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)
p = Expression(("0", "0", "p0*cos(t/tc)"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

# Define function space for displacement, velocity and acceleration
V = VectorFunctionSpace(mesh, "CG", 1)

# Define function space for stresses
Vsig = TensorFunctionSpace(mesh, "DG", 0)

# Test and trial functions
du = TrialFunction(V)
u_ = TestFunction(V)
# Current (unknown) displacement
u = Function(V, name="Displacement")
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)

# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(right)
force_boundary.mark(boundary_subdomains, 3)

# Define measure for boundary condition integral
dss = ds(subdomain_data=boundary_subdomains)

# Set up boundary condition at left end
zero = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, zero, left)

# Stress tensor
def sigma(r):
    return 2.0*mu*sym(grad(r)) + lmbda*tr(sym(grad(r)))*Identity(len(r))

# Mass form
def m(u, u_):
    return rho*inner(u, u_)*dx

# Elastic stiffness form
def k(u, u_):
    return inner(sigma(u), sym(grad(u_)))*dx

# Rayleigh damping form
def c(u, u_):
    return eta_m*m(u, u_) + eta_k*k(u, u_)

# Work of external forces
def Wext(u_):
    return dot(u_, p)*dss(3)

# Update formula for acceleration
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u, u_old, v_old, a_old):
    u_vec, u0_vec  = u.vector(), u_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    u_old.vector()[:] = u.vector()

def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new

a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)
res = m(avg(a_old, a_new, alpha_m), u_) + c(avg(v_old, v_new, alpha_f), u_) \
       + k(avg(u_old, du, alpha_f), u_) - Wext(u_)
a_form = lhs(res)
L_form = rhs(res)

# Define solver for reusing factorization
K, res = assemble_system(a_form, L_form, bc)
solver = LUSolver(K, "mumps")
solver.parameters["symmetric"] = True


# Time-stepping
time = np.linspace(0, T, Nsteps+1)
u_tip = np.zeros((Nsteps+1,))
energies = np.zeros((Nsteps+1, 4))
E_damp = 0
E_ext = 0
sig = Function(Vsig, name="sigma")


################################################################### time loop
from vtkplotter.dolfin import *

# add a frame box
box = Box([2., .57, -0.0], 8, 3, 2).wireframe().alpha(0)
flogo = load(datadir+'images/fenics_logo.png').scale(.004).pos(-1.5,-0.1,-.0)

vp = Plotter(bg='w', axes=0, interactive=0)

vp.camera.SetPosition( [-0.493, 7.837, 15.111] )
vp.camera.SetFocalPoint( [2.0, 0.57, 0.0] )
vp.camera.SetViewUp( [0.031, 0.903, -0.429] )
vp.camera.SetDistance( 16.952 )
vp.camera.SetClippingRange( [12.558, 22.524] )

#vd = Video()

pb = ProgressBar(0, len(np.diff(time)), c='blue')
for (i, dt) in enumerate(np.diff(time)):
    t = time[i+1]

    # Forces are evaluated at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt
    p.t = t-float(alpha_f*dt)

    # Solve for new displacement
    res = assemble(L_form)
    bc.apply(res)
    solver.solve(K, u.vector(), res)

    update_fields(u, u_old, v_old, a_old)
    p.t = t
    
    #####################################
    act = MeshActor(u).wireframe(False)
    coords = act.coordinates()
    act.pointColors(coords[:,2], cmap='afmhot_r', vmin=0.05)
    act.move()
    vp.show(act, box, flogo)
    screenshot('ss'+str(i)+'.png')
#    vd.addFrame()
    pb.print()

#vd.close()

