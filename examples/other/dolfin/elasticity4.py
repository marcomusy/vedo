import numpy as np
from dolfin import *
import vedo

set_log_level(30)


class AllBoundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]<-10

def solve_problem(mesh, mfunc, force):
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    displacement = Function(V)

    bc = [DirichletBC(V, Constant((0,0)), mfunc, 1)]

    F = Constant(force)
    E = Constant(5000)
    nu = Constant(0.3)
    mu = E / (2.0*(1+nu))
    lmbda = E * nu / ((1.0+nu) * (1-2*nu))
    sigma = 2.0 * mu * sym(grad(u)) + lmbda * tr( sym(grad(u)) ) * Identity(2)
    solve(inner(sigma, grad(v)) * dx == inner(F, v) * dx, displacement, bc)
    displacement.set_allow_extrapolation(True)
    return displacement

def update(mesh, displacement):
    new_mesh = Mesh(mesh)
    ALE.move(new_mesh, displacement)
    return new_mesh

def remesh(mesh):
    if isinstance(mesh, vedo.Mesh):
        vmesh = mesh
    else:
        vmesh = vedo.Mesh([mesh.coordinates(), mesh.cells()])
    bpts = vmesh.compute_normals(cells=True).boundaries().join(reset=1) #extract boundary
    vz = vmesh.celldata["Normals"][0][2] # check z component of normals at first point
    bpts.generate_mesh(invert=vz<0).write('tmpmesh.xml') #vedo REMESHING + smoothing
    return Mesh("tmpmesh.xml")

#################################################################################
N = 20         # number of iterations of stretching
do_remesh = 0  # grab the boundary and remesh the interior at each iteration

circle = vedo.Circle(r=50)
mesh = remesh(circle)
half_circle = circle.boundaries().cut_with_plane(origin=[-10,0,0], normal='-x').z(2)
half_circle.linewidth(5).c("red4")

plt = vedo.Plotter(N=N, size=(2250, 1300))

meshes = [mesh]
displacements = []
for i in range(N):
    mfunc = MeshFunction('size_t', mesh, 1, mesh.domains())
    mfunc.set_all(0)
    allb = AllBoundaries()
    allb.mark(mfunc, 1)

    F = np.array([4, 2*(i-N/2)/N])
    displacement = solve_problem(mesh, mfunc, F)
    new_mesh = update(mesh, displacement)

    mesh = remesh(new_mesh) if do_remesh else new_mesh
    meshes.append(mesh)
    displacements.append(displacement)

    varrow = vedo.Arrow2D([0,0], F*15).z(1).c("red4")
    vmesh = vedo.Mesh([mesh.coordinates(), mesh.cells()]).c("k4").lc('k5')
    plt.at(i).show(f"t={i}, F={F}", half_circle, vmesh, varrow, zoom=1.5)

plt.interactive().close()


