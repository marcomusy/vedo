"""Find the closest point
on the mesh to each random point
"""
import trimesh
import numpy as np
from vedo import show, Arrows, download

plyfile = download('https://github.com/mikedh/trimesh/blob/main/models/cycloidal.ply')

mesh = trimesh.load(plyfile)
points = mesh.bounding_box_oriented.sample_volume(count=30)

# find the closest point on the mesh to each random point
closest_points, distances, triangle_id = mesh.nearest.on_surface(points)
#print('Distance from point to surface of mesh:\n{}'.format(distances))

# create a PointCloud object out of each (n,3) list of points
cloud_original = trimesh.points.PointCloud(points)
cloud_close    = trimesh.points.PointCloud(closest_points)

# create a unique color for each point
cloud_colors = np.array([trimesh.visual.random_color() for i in points])

# set the colors on the random point and its nearest point to be the same
cloud_original.vertices_color = cloud_colors
cloud_close.vertices_color    = cloud_colors

arrs = Arrows(cloud_original.vertices, cloud_close.vertices, c='w')

## create a scene containing the mesh and two sets of points
show(mesh, cloud_original, cloud_close, arrs, __doc__, bg='bb', axes=1, viewup='z').close()
