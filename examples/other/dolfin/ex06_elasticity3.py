#!/usr/bin/env python3
"""An initial circle is stretched by means of a variable force into its final shape.
Colored lines are the trajectories of a few initial points."""
from dolfin import *
from mshr import *
import numpy as np
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
    lmbda = E * nu / ((1.0+nu) * (1.0-2*nu))
    sigma = 2.0 * mu * sym(grad(u)) + lmbda * tr( sym(grad(u)) ) * Identity(2)
    solve(inner(sigma, grad(v)) * dx == inner(F, v) * dx, displacement, bc)
    displacement.set_allow_extrapolation(True)
    return displacement

def update(mesh, displacement):
    new_mesh = Mesh(mesh)
    ALE.move(new_mesh, displacement)
    return new_mesh

def remesh(mesh, res=50):
    if isinstance(mesh, vedo.Mesh):
        vmesh = mesh
    else:
        vmesh = vedo.Mesh([mesh.coordinates(), mesh.cells()])
    bpts = vmesh.computeNormals(cells=True).boundaries().join(reset=1) #extract boundary
    vz = vmesh.celldata["Normals"][0][2] # check z component of normals at first point
    bpts.tomesh(invert=vz<0).smooth().write('tmpmesh.xml') #vedo REMESHING + smoothing
    return Mesh("tmpmesh.xml")

#################################################################################
N = 40             # number of iterations of stretching
res = 15           # resolution of meshes
do_remesh = False  # grab the boundary and remesh the interior at each iteration
vedo.settings.useParallelProjection = True  # avoid perspective parallax

circle = Circle(Point(0, 0), 50)
mesh = generate_mesh(circle, res)

meshes = [mesh]
displacements = []
for i in range(N):
    mfunc = MeshFunction('size_t', mesh, 1, mesh.domains())
    mfunc.set_all(0)
    allb = AllBoundaries()
    allb.mark(mfunc, 1)

    F = np.array([2, (i-N/2)/N]) # some variable force

    displacement = solve_problem(mesh, mfunc, F)
    new_mesh = update(mesh, displacement)

    if do_remesh:
        mesh = remesh(new_mesh)
    else:
        mesh = new_mesh
    meshes.append(mesh)
    displacements.append(displacement)
    # plot things:
    txt = vedo.Text2D(f"step{i}")
    arrow = vedo.Arrow2D([0,0], F*20).z(1)
    vedo.dolfin.plot(mesh, arrow, txt, c='grey5', at=i, N=N, zoom=1.1) #PRESS q

dmesh_i = meshes[0]  # initial mesh
dmesh_f = meshes[-1] # final mesh

vmesh_i = vedo.Mesh([dmesh_i.coordinates(), dmesh_i.cells()], c='grey5').z(-1)
vmesh_f = vedo.Mesh([dmesh_f.coordinates(), dmesh_f.cells()], c='grey3').wireframe()

plt = vedo.Plotter()

# move a few points along the deformation of the circle
seeds = vedo.Circle(r=50, res=res).points()[:,(0,1)] # make points 2d with [:,(0,1)]
endpoints = []
for i, p in enumerate(seeds):
    line = [p]
    for u in displacements:
        p = p + u(p)
        line.append(p)
    plt += vedo.Line(line, c=i, lw=4).z(1)
    endpoints.append(p)

plt += [vmesh_i, vmesh_f, __doc__]
plt.show(axes=1)

# to invert everything and move the end points back in place, check out discussion:
#https://fenicsproject.discourse.group/t/precision-in-hyperelastic-model/6824/3


