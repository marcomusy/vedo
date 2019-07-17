"""
Show fenics mesh and displacement solution.
"""
# Refer to original script for the details:
# https://fenicsproject.org/docs/dolfin/2018.1.0/python/
#        demos/hyperelasticity/demo_hyperelasticity.py.html
#
print(__doc__)

########################################################### dolfin
from dolfin import *

# Create mesh and define function space
mesh = UnitCubeMesh(12, 12, 12)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomains
left  = CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side=1.0)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Constant((0.0, 0.0, 0.0))
r = Expression((
        "scale*0.0",
        "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
        "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])",
    ),
    scale=0.5,
    y0=0.5,
    z0=0.5,
    theta=pi/4,
    degree=2,
)

bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, r, right)

w = TrialFunction(V)  # Incremental displacement
v = TestFunction(V)  # Test function
u = Function(V)  # solution

solve(inner(grad(w), grad(v)) * dx == inner(c, v) * dx, u, [bcl, bcr])


########################################################### vtkplotter
from vtkplotter.dolfin import *
from vtkplotter import mag

# Create the objects to be visualised as MeshActor(vtkActor)
pts0 = MeshPoints(mesh)  # get the original points from the mesh

pts1 = MeshActor(mesh).alpha(0.4)  # representation with mesh faces.
pts1.move(u)  # deform mesh actor according to solution u

# Open N=2 synced renderers and draw on the first one (at=0)
show(pts0, pts1, at=0, N=2, depthpeeling=True)


# Reuse same solution u but on a different, finer mesh:
mesh_big = UnitCubeMesh(15, 15, 15)

arrs = MeshArrows(mesh_big, u)

# Draw the result on the second, and allow user interaction:
show(arrs, at=1)  # can use interactive=1 to hold the plot

#########################
# Let's open a new window and draw the same stuff with colors
scl = mag(pts1.coordinates() - pts0.coordinates())  # size of displacements

# make a working copy with clone() and colorize vertices with scalars
pts3 = pts1.clone().wireframe(False).pointColors(scl, cmap="rainbow")

show(pts3, pos=(200, 200), depthpeeling=True, newPlotter=True, axes=8)
# (Try click the mesh and press shift-X to slice it)
