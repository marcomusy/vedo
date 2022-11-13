"""Generate a Volume with the signed distance from a Mesh,
then generate the isosurface at distance -0.5"""
from vedo import *

mesh = Mesh(dataurl+"beethoven.ply").subdivide()
mesh.color('k').point_size(3) # render mesh as points

# Generate signed distance volume
vol = mesh.signed_distance(dims=(40,40,40))

# Generate an isosurface at distance -0.5
iso = vol.isosurface(-0.5)

show(mesh, iso, __doc__, axes=1).close()
