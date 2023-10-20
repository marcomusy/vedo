import numpy as np
import napari
import vedo

print("\nSEE ALSO https://github.com/jo-mueller/napari-vedo-bridge")

# Load the surface, triangulate just in case, and compute vertex normals
surf = vedo.Mesh(vedo.dataurl+"beethoven.ply").triangulate().compute_normals()
surf.rotate_x(180).rotate_y(60)

vertices = surf.vertices
faces    = np.array(surf.cells)
normals  = surf.vertex_normals
# generate vertex values by projecting normals on a "lighting vector"
values   = np.dot(normals, [-1, 1, 1])

# create an empty viewer
viewer = napari.Viewer()

# add the surface
viewer.add_surface((vertices, faces, values), opacity=0.8)
viewer.add_points(vertices, size=0.05, face_color='pink')

# turn on 3D rendering
viewer.dims.ndisplay = 3
napari.run()