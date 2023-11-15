"""Remove points and cells from a mesh
which are closest to a specified point."""
from vedo import *

# Enable depth peeling for the scene
settings.use_depth_peeling = True

# Load the apple mesh from a url, set the colors and line width
msh = Mesh(dataurl+'apple.ply')
msh.c('lightgreen').bc('tomato').lw(1)

# Set a point and a radius to find the closest points in the mesh to it
pt = [1, 0.5, 1]
R = 1.2
ids = msh.closest_point(pt, radius=R, return_point_id=True)

# Remove the cells from the mesh by their ids
# and clean the mesh by removing orphaned vertices not associated to any cell
printc('#points before:', msh.npoints, c='g')
msh.delete_cells_by_point_index(ids)
msh.clean()
printc('#points after :', msh.npoints, c='g')

# Create a sphere object with the given point and radius, and set transparency
sph = Sphere(pt, r=R, alpha=0.1)

# Show the point, the sphere, the modified mesh, the script docstring and axes
# then close the window
show(Point(pt), sph, msh, __doc__, axes=1).close()
