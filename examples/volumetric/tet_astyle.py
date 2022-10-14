"""Visualize a TetMesh with
default ray casting."""
from vedo import *

# settings.use_depth_peeling = False

tetm = TetMesh(dataurl+'limb_ugrid.vtk')
tetm.color('jet').alpha_unit(100) # make the tets more transparent
tetm.add_scalarbar3d()

# Build a Mesh object made of all the boundary triangles
wmesh = tetm.tomesh(fill=False).wireframe()

# Make a copy of tetm and shrink the tets
shrunk = tetm.clone().shrink(0.5)

# Build a Mesh object and cut it
cmesh = shrunk.tomesh(fill=True)

show([(tetm, __doc__),
      (wmesh, "..wireframe surface"),
      (cmesh, "..shrunk tetrahedra"),
     ], N=3, axes=1,
).close()

