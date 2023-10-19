"""Cut a TetMesh with an arbitrary polygonal Mesh"""
from vedo import *

tetmesh = TetMesh(dataurl+'limb_ugrid.vtk')

sphere = Sphere(r=500, c='green5', alpha=0.2).x(400)

ugrid = tetmesh.cut_with_mesh(sphere, invert=True)
tetmesh_cut = TetMesh(ugrid)
print(tetmesh_cut)

show(
    tetmesh_cut.tomesh(),
    sphere,
    __doc__,
    axes=dict(xtitle='x [:mum]'),
).close()

