import numpy as np
import napari, vedo

# Load the surface, triangulate just in case, and compute vertex normals
surf = vedo.Mesh(vedo.dataurl+"beethoven.ply").triangulate().computeNormals()
surf.rotateX(180).rotateY(60)

vertices = surf.points()
faces    = np.array(surf.faces())
normals  = surf.normals()
# generate vertex values by projecting normals on a "lighting vector"
values   = np.dot(normals, [-1, 1, 1])

print(vertices.shape, faces.shape, values.shape)
# (2521, 3) (5030, 3) (2521,)

with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer()

    # add the surface
    viewer.add_surface((vertices, faces, values), opacity=0.8)
    viewer.add_points(vertices, size=0.05, face_color='pink')

    # turn on 3D rendering
    viewer.dims.ndisplay = 3