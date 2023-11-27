"""Cut a TetMesh with a Mesh 
to generate an UnstructuredGrid"""
from vedo import *

settings.default_font = 'Calco'

sphere = Sphere(r=500).x(400).c('green', 0.1)

tetm1 = TetMesh(dataurl+'limb.vtu')
tetm1.cmap('jet', tetm1.vertices[:, 2], name="ProximoDistal")

# Clone and cut the TetMesh, this returns a UnstructuredGrid:
ugrid1 = tetm1.clone().cut_with_mesh(sphere, invert=True)
ugrid1.cmap("Purples_r", "SignedDistance")
print(ugrid1)

# Cut tetm, but the output will keep only the whole tets (NOT the polygonal boundary!):
ugrid2 = tetm1.clone().cut_with_mesh(sphere, invert=True, whole_cells=True)
tetm2 = TetMesh(ugrid2).cmap("Greens_r", "ProximoDistal")
print(tetm2)

# Cut tetm, but the output will keep only the tets on the boundary:
ugrid3 = tetm1.clone().cut_with_mesh(sphere, on_boundary=True)
tetm3 = TetMesh(ugrid3)
tetm3.celldata.select("chem_0").cmap("Reds")
print(tetm3)

show([
      (ugrid1,sphere,  __doc__),
      (tetm2, sphere, "Keep only tets that lie\ncompletely outside of the Sphere"),
      (tetm3, sphere, "Keep only tets that lie\nexactly on the Sphere"),
     ], 
     N=3, axes=dict(xtitle='x in :mum'),
).close()
