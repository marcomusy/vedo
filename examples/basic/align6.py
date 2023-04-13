
"""Align to bounding boxes. Force the Mesh into the empty box."""
from vedo import *

# Load a mesh and color it silver
msh1 = Mesh(dataurl + "cessna.vtk").color("silver")

# Create axes for the original mesh
axes1 = Axes(msh1)

# Create a wireframe cube at a specified position
cube = Cube().pos(2, 1, 0).wireframe()

# Clone the mesh and align it to the bounding box of the cube
msh2 = msh1.clone().align_to_bounding_box(cube)

# Create axes for the aligned mesh
axes2 = Axes(msh2)

# Set up a Plotter object with 2 subrenderers
plt = Plotter(N=2)

# Show the original mesh, axes, and cube in the left renderer with the script description
plt.at(0).show(msh1, axes1, cube, __doc__)

# Show the aligned mesh and axes in the right renderer, viewing from the top
plt.at(1).show(msh2, axes2, viewup='z')

# Enable interaction and close the plotter when done
plt.interactive().close()
