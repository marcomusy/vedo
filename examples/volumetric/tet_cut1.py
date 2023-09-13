"""Cut a TetMesh with an arbitrary polygonal Mesh"""
from vedo import *

settings.use_depth_peeling = True

tetmesh = TetMesh(dataurl+'limb_ugrid.vtk')

sphere = Sphere(r=500, c='g').x(400).alpha(0.2)

ugrid = tetmesh.cut_with_mesh(sphere, invert=True)
tetmesh_cut = TetMesh(ugrid)

show(tetmesh_cut, sphere, __doc__, axes=dict(xtitle='x [:mum]')).close()
