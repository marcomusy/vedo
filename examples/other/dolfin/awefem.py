"""
Solve the constant velocity scalar wave equation in an arbitrary number of dimensions.
It injects a point source with a time-dependent source time function.
"""
#Original script by Carlos da Costa:
#https://github.com/cako/fenics-scripts/blob/master/awefem/awefem.py
#
from dolfin import *
from vedo import settings
from vedo.dolfin import plot, interactive, ProgressBar, printc, download
import numpy as np

set_log_level(30)
settings.allowInteraction = True

def ricker_source(t, f=40):
    t -= 2 / f
    return (1 - 2 * (np.pi*f*t)**2) * np.exp(-(np.pi*f*t)**2)

def sine_source(t, f=40):
    return np.sin(2 * np.pi*f*t)


def awefem(mesh, t, source_loc=None):

    # Function space
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Boundary condition
    bc = DirichletBC(V, Constant(0), "on_boundary")

    # Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Discretization
    c = 6
    dt = t[1] - t[0]
    u0 = Function(V)  # u0 = uN-1
    u1 = Function(V)  # u1 = uN1

    # Variational formulation
    F = (u - 2 * u1 + u0) * v * dx + (dt * c) ** 2 * dot(
        grad(u + 2 * u1 + u0) / 4, grad(v) ) * dx
    a, L = lhs(F), rhs(F)

    # Solver
    A, b = assemble_system(a, L)
    solver = LUSolver(A, "mumps")
    solver.parameters["symmetric"] = True
    bc.apply(A, b)

    # Solution
    u = Function(V)  # uN+1

    # Source
    if source_loc is None:
        mesh_center = np.mean(mesh.coordinates(), axis=0)
        source_loc = Point(mesh_center)
    else:
        source_loc = Point(source_loc)

    # Time stepping
    printc('\bomb Hit F1 to interrupt.', c='yellow')
    pb = ProgressBar(0, len(t))
    for i, t_ in enumerate(t[1:]):
        pb.print()
        b = assemble(L)
        delta = PointSource(V, source_loc, ricker_source(t_) * dt**2)
        delta.apply(b)
        solver.solve(u.vector(), b)

        u0.assign(u1)
        u1.assign(u)

        if t_>0.03:
            plot(u,
                 warpZfactor=20, # set elevation along z
                 vmin=.0,     # sets a minimum to the color scale
                 vmax=0.003,
                 cmap='rainbow', # the color map style
                 alpha=1,        # transparency of the mesh
                 lw=0.1,         # linewidth of mesh
                 scalarbar=None,
                 #lighting='plastic',
                 #elevation=-.3,
                 interactive=0)  # continue execution

    interactive()

if __name__ == "__main__":

    ot, dt, nt = 0, 1e-3, 150
    t = ot + np.arange(nt) * dt

    print("Computing wavefields over dolfin mesh")
    fpath = download("https://vedo.embl.es/examples/data/dolfin_fine.xml")
    mesh = Mesh(fpath)
    awefem(mesh, t, source_loc=(0.8, 0.8))

#    print('Computing wavefields over unit square')
#    mesh = UnitSquareMesh(100, 100)
#    u = awefem(mesh, t, source_loc=(0.8, 0.7))

#    print('Computing wavefields over unit circle')
#    domain = Circle(Point(0., 0.), 1)
#    mesh = generate_mesh(domain, 50)
#    u = awefem(mesh, t, source_time_function=sine_source)

#    print('Computing wavefields over unit cube')
#    print('need to set alpha=0.1 and warpZfactor=0')
#    mesh = UnitCubeMesh(15, 15, 15)
#    u = awefem(mesh, t, source_loc=(0.8, 0.7, 0.7))
