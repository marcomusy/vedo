"""Cut a TetMesh with
an arbitrary polygonal Mesh
Units are \mum"""
from vedo import *

settings.useDepthPeeling = True

tetmesh = TetMesh(dataurl+'limb_ugrid.vtk')

sphere = Sphere(r=500, c='g').x(400).alpha(0.2)

tetmesh.cutWithMesh(sphere, invert=True)

show(tetmesh, sphere, __doc__, axes=dict(xtitle='x [\mum]')).close()
