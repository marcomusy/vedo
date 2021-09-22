"""Cut a TetMesh with a Mesh
(note the presence of polygonal boundary)"""
from vedo import *

settings.useDepthPeeling = True

tetm = TetMesh(dataurl+'limb_ugrid.vtk')

sphere = Sphere(r=500).x(400).c('green', 0.1)

# Clone and cut tetm, keep the outside:
tetm1 = tetm.clone().cutWithMesh(sphere, invert=True)

# Make it a polygonal Mesh for visualization
msh1 = tetm1.tomesh().lineWidth(0.1).color('lb')

# Cut tetm, but the output will keep only the whole tets (NOT the polygonal boundary!):
tetm2 = tetm.clone().cutWithMesh(sphere, invert=True, wholeCells=True)

# Cut tetm, but the output will keep only the tets on the boundary:
tetm3 = tetm.clone().cutWithMesh(sphere, onlyBoundary=True)
tetm3.addScalarBar3D(c='k')

show([(msh1, sphere, __doc__),
      (tetm2.tomesh(), "Keep only tets that lie\ncompletely outside the Sphere"),
      (tetm3.tomesh(), sphere, "Keep only tets that lie\nexactly on the Sphere"),
     ], N=3, axes=dict(xtitle='x in \mum')).close()
