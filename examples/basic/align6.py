"""Align to bounding boxes. Force the Mesh into the box."""
from vedo import *

msh1 = Mesh(dataurl+"cessna.vtk").color("silver")
axes1 = Axes(msh1)

cube = Cube().pos(2,1,0).wireframe()

msh2 = msh1.clone().alignToBoundingBox(cube)
axes2 = Axes(msh2)

plt = Plotter(N=2)
plt.at(0).show(msh1, axes1, cube, __doc__)
plt.at(1).show(msh2, axes2, viewup='z')
plt.interactive().close()
