"""Generate a scalar field by the signed distance from a mesh,
optionally save it to a vti file,
then extract an isosurface from the 3d image.
"""
from vtkplotter import *

mesh = load(datadir+"pumpkin.vtk")

# Generate signed distance volume
vol = volumeFromMesh(mesh,
                     dims=(40,40,40),
                     bounds=(-1.1, 1.1, -1.1, 1.1, -1.1, 1.1),
                     signed=True,
                     negate=True, # invert sign
)
#write(vol, 'stack.vti')

iso = isosurface(vol, threshold=-0.01)

pts = Points(mesh.coordinates())

show(iso, pts, Text(__doc__), axes=1)
