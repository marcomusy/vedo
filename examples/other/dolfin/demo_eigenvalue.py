# A simple eigenvalue solver
# ==========================

from dolfin import *
from vedo.dolfin import download, plot

# Define mesh, function space
fpath = download("https://vedo.embl.es/examples/data/box_with_dent.xml.gz")
mesh = Mesh(fpath)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx

# Assemble stiffness form
A = PETScMatrix()
assemble(a, tensor=A)

# Create eigensolver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve()

# Extract largest (first) eigenpair
r, c, rx, cx = eigensolver.get_eigenpair(0)
print("Largest eigenvalue: ", r)

# Initialize function and assign eigenvector
u = Function(V)
u.vector()[:] = rx

# plot eigenfunction on mesh as colored points (ps=point size)
plot(u, mode='mesh', ps=12, cmap='gist_earth')

#or as wireframe
plot(u, mode='mesh', wireframe=True, cmap='magma')
