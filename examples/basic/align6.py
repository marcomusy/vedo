"""Align to bounding boxes. Force the Mesh into the empty box."""
from vedo import Mesh, dataurl, Axes, Cube, Plotter

# Reference mesh.
msh1 = Mesh(dataurl + "cessna.vtk").color("silver")
axes1 = Axes(msh1)

# Target bounding box.
cube = Cube().pos(2, 1, 0).wireframe()

# Align mesh to the cube bounding box.
msh2 = msh1.clone().align_to_bounding_box(cube)
axes2 = Axes(msh2)

plt = Plotter(N=2)
plt.at(0).add(msh1, axes1, cube, __doc__)
plt.at(1).add(msh2, axes2, cube)

plt.show(viewup="z", zoom=0.6)
plt.interactive().close()
