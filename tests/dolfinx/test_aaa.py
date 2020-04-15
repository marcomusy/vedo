# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This demo solves the equations of static linear elasticity for a
# pulley subjected to centripetal accelerations. The solver uses
# smoothed aggregation algebraic multigrid.

from contextlib import ExitStack

import numpy as np
from petsc4py import PETSc

import dolfinx
from dolfinx import (MPI, UnitCubeMesh, 
                     DirichletBC, Function, VectorFunctionSpace, cpp)
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import apply_lifting, assemble_matrix, assemble_vector, set_bc
#from dolfin.io import XDMFFile
from dolfinx.la import VectorSpaceBasis
from ufl import (Identity, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, dx, grad, inner, sym, tr)


def build_nullspace(V):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    index_map = V.dofmap.index_map
    nullspace_basis = [cpp.la.create_vector(index_map) for i in range(6)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        # Build translational null space basis
        V.sub(0).dofmap.set(basis[0], 1.0)
        V.sub(1).dofmap.set(basis[1], 1.0)
        V.sub(2).dofmap.set(basis[2], 1.0)

        # Build rotational null space basis
        V.sub(0).set_x(basis[3], -1.0, 1)
        V.sub(1).set_x(basis[3], 1.0, 0)
        V.sub(0).set_x(basis[4], 1.0, 2)
        V.sub(2).set_x(basis[4], -1.0, 0)
        V.sub(2).set_x(basis[5], 1.0, 1)
        V.sub(1).set_x(basis[5], -1.0, 2)

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(6)]
    nsp = PETSc.NullSpace()
    nsp.create(_x)
    return nsp


# Load mesh from file
# mesh = Mesh(MPI.comm_world)
# XDMFFile(MPI.comm_world, "../pulley.xdmf").read(mesh)

mesh = UnitCubeMesh(MPI.comm_world, 3, 3, 3)
#mesh = BoxMesh(
#    MPI.comm_world, [np.array([0.0, 0.0, 0.0]),
#                     np.array([2.0, 1.0, 1.0])], [12, 12, 12],
#    CellType.tetrahedron, dolfin.cpp.mesh.GhostMode.none)
cmap = dolfinx.fem.create_coordinate_map(mesh.ufl_domain())
mesh.geometry.coord_mapping = cmap

def boundary(x):
    return np.logical_or(x[0] < 10.0 * np.finfo(float).eps,
                         x[0] > 1.0 - 10.0 * np.finfo(float).eps)


# Rotation rate and mass density
omega = 300.0
rho = 10.0

# Loading due to centripetal acceleration (rho*omega^2*x_i)
x = SpatialCoordinate(mesh)
f = as_vector((rho * omega**2 * x[0], rho * omega**2 * x[1], 0.0))

# Elasticity parameters
E = 1.0e9
nu = 0.0
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def sigma(v):
    return 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(
        len(v))


# Create function space
V = VectorFunctionSpace(mesh, ("Lagrange", 1))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v)) * dx
L = inner(f, v) * dx

u0 = Function(V)
with u0.vector.localForm() as bc_local:
    bc_local.set(0.0)

# Set up boundary condition on inner surface
bc = DirichletBC(V, u0, boundary)

# Assemble system, applying boundary conditions and preserving symmetry)
A = assemble_matrix(a, [bc])
A.assemble()

b = assemble_vector(L)
apply_lifting(b, [a], [[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Create solution function
u = Function(V)

# Create near null space basis (required for smoothed aggregation AMG).
null_space = build_nullspace(V)

# Attach near nullspace to matrix
A.setNearNullSpace(null_space)

# Set solver options
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-12
opts["pc_type"] = "gamg"

# Use Chebyshev smoothing for multigrid
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

# Improve estimate of eigenvalues for Chebyshev smoothing
opts["mg_levels_esteig_ksp_type"] = "cg"
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

# Create CG Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(MPI.comm_world)
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)

# Compute solution
solver.setMonitor(lambda ksp, its, rnorm: print("Iteration: {}, rel. residual: {}".format(its, rnorm)))
solver.solve(b, u.vector)
#solver.view()




############################### Plot solution
from vtkplotter.dolfin import plot

plot(u, mode="displaced mesh",
     scalarbar=False,
     axes=1,
     bg='white',
     viewup='z',
     offscreen=1)

#################################################################################
from vtkplotter import settings, screenshot
actor = settings.plotter_instance.actors[0]
solution = actor.scalars(0)

screenshot('elasticbeam.png')

print('ArrayNames', actor.getArrayNames())
print('min', 'mean', 'max, N:')
print(np.min(solution), np.mean(solution), np.max(solution), len(solution))

# Plot solution
# import matplotlib.pyplot as plt
# import dolfin.plotting
# dolfin.plotting.plot(u)
# plt.show()