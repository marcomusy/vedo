"""Generate a Volume with the signed distance from a Mesh,
then generate the isosurface at distance -0.5"""
from vedo import *

mesh = Mesh(dataurl+"beethoven.ply").subdivide()
mesh.color('k').pointSize(3) # render mesh as points

# Generate signed distance volume
vol = mesh.signedDistance(dims=(40,40,40))

# Generate an isosurface at distance -0.5
iso = vol.isosurface(threshold=-0.5)

show(mesh, iso, __doc__, axes=1).close()
