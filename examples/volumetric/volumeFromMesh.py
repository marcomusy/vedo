"""Generate a scalar field by the signed distance from a mesh,
optionally save it to a vti file,
then extract an isosurface from the 3d image."""
from vedo import *

mesh = Mesh(dataurl+"apple.ply").subdivide()

# Generate signed distance volume
vol = volumeFromMesh(mesh,
                     dims=(40,40,40),
                     bounds=(-1.3, 1.3, -1.3, 1.3, -1.3, 1.3),
                     signed=True,
                     negate=True, # invert sign
)
#write(vol, 'stack.vti')

iso = vol.isosurface(threshold=-0.01)

pts = Points(mesh.points())

show(iso, pts, __doc__, axes=1).close()
