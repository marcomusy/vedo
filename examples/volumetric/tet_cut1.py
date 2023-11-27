"""Cut a TetMesh with an arbitrary polygonal Mesh"""
from vedo import *

sphere = Sphere(r=500).x(400)
sphere.color('green5', 0.2).wireframe()

tmesh = TetMesh(dataurl + 'limb.vtu')
print(tmesh)

ugrid = tmesh.cut_with_mesh(sphere, invert=True).cmap("Reds_r")
print(ugrid)

# We may cast the output to a new TetMesh:
# tmesh_cut = TetMesh(ugrid)
# print(tmesh_cut)

show(ugrid, sphere, __doc__, axes=1).close()
