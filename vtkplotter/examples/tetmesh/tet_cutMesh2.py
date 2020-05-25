"""Cut a TetMesh with a Mesh"""
from vtkplotter import *

tetmesh = TetMesh(datadir+'limb_ugrid.vtk')

sphere = Sphere(r=500).x(400).c('green').alpha(0.1)

# Clone and cut tetmesh:
tetmesh1 = tetmesh.clone().cutWithMesh(sphere, invert=True)

# Make it a polygonal Mesh for visualization
msh1 = tetmesh1.toMesh().lineWidth(0.1).color('lb')
msh1.addScalarBar3D()

# Cut tetmesh, but the output will keep only the tets (NOT the polygonal boundary!):
tetmesh2 = tetmesh.clone().cutWithMesh(sphere, invert=True, onlyTets=True)

# Cut tetmesh, but the output will keep only the tets on the boundary:
tetmesh3 = tetmesh.clone().cutWithMesh(sphere, onlyBoundary=True)

show([(msh1, sphere, __doc__),
      (tetmesh2.toMesh(), "Keep only tets that lie\ncompletely outside the Sphere"),
      (tetmesh3.toMesh(), sphere, "Keep only tets that lie\nexactly on the Sphere"),
     ], N=3, axes=1)
