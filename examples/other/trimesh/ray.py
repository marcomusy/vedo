import trimesh
import numpy as np
from vedo import show, settings

settings.useDepthPeeling = True

# test on a sphere mesh
mesh = trimesh.creation.icosphere()

# create some rays
ray_origins    = np.array([[0, 0, -3], [1,  2, -3]])
ray_directions = np.array([[0, 0,  1], [0, -1,  1]])

# run the mesh-ray query
locations, index_ray, index_tri = mesh.ray.intersects_location(
    ray_origins=ray_origins, ray_directions=ray_directions
)
locs = trimesh.points.PointCloud(locations)

# stack rays into line segments for visualization as Path3D
ray_visualize = trimesh.load_path(
    np.hstack((ray_origins, ray_origins + ray_directions)).reshape(-1, 2, 3)
)

print("The rays hit the mesh at coordinates:\n", locations)
print("The rays with index: {} hit triangles stored at mesh.faces[{}]".format(index_ray, index_tri))

# stack rays into line segments for visualization as Path3D
ray_visualize = trimesh.load_path(
    np.hstack((ray_origins, ray_origins + ray_directions * 5.0)).reshape(-1, 2, 3)
)

# make mesh white-ish
mesh.visual.face_colors = [200, 200, 250, 100]
mesh.visual.face_colors[index_tri] = [255, 0, 0, 255]

show(mesh, ray_visualize, locs, axes=1).close()
