"""Visualize a TetMesh with
default ray casting."""
from vedo import *

# settings.useDepthPeeling = False

tetm = TetMesh(dataurl+'limb_ugrid.vtk')
tetm.color('jet').alphaUnit(100) # make the tets more transparent
tetm.addScalarBar3D()

# Build a Mesh object made of all the boundary triangles
wmesh = tetm.tomesh(fill=False).wireframe()

# Make a copy of tetm and shrink the tets
shrinked = tetm.clone().shrink(0.5)

# Build a Mesh object and cut it
cmesh = shrinked.tomesh(fill=True)

show([(tetm, __doc__),
      (wmesh, "..wireframe surface"),
      (cmesh, "..shrinked tetrahedra"),
     ], N=3, axes=1,
).close()

