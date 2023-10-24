"""Find cells within specified bounds in x, y and/or z."""
from vedo import *

# Load a mesh of a shark and normalize it
mesh = Mesh(dataurl+'shark.ply').normalize().compute_normals()

# Set the color of the mesh and the line width to 1
mesh.color('aqua').linewidth(1)

# Define the lower and upper bounds for the z-axis
z1, z2 = -1.5, -0.5

# Find the cell IDs of cells within the z-axis bounds
ids = mesh.find_cells_in_bounds(zbounds=(z1,z2))

# Print the cell IDs in green to the console
printc('IDs of cells within bounds:\n', sorted(ids), c='g')

# Create two Plane objects at the specified z-positions
p1 = Plane(normal=(0,0,1), s=[2,2]).z(z1).alpha(0.5)
p2 = p1.clone().z(z2)

# Set the color of cells within the bounds to red
mesh.cellcolors[ids] = [200,10,10, 255] #RGBA 

labels = mesh.labels("cellid", scale=0.01)

# Show the mesh, the two planes, the docstring
show(mesh, p1, p2, labels, __doc__, axes=1).close()
