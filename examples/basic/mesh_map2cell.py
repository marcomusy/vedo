"""Map an array which is defined on
the vertices of a mesh to its cells"""
from vedo import *

doc = Text2D(__doc__, pos="top-center")

mesh1 = Mesh(dataurl+'icosahedron.vtk').lineWidth(0.1).flat()

# let the scalar be the z coordinate of the mesh vertices
msg1 = Text2D("Scalars originally defined on points..", pos="bottom-center")
mesh1.pointdata["myzscalars"] = mesh1.points()[:, 2]

mesh1.cmap("jet", "myzscalars", on="points")

msg2 = Text2D("..are interpolated to cells.", pos="bottom-center")
mesh2 = mesh1.clone(deep=False).mapPointsToCells()

plt = Plotter(N=2, axes=11)
plt.at(0).show(mesh1, msg1, doc, viewup="z")
plt.at(1).show(mesh2, msg2)
plt.interactive().close()
