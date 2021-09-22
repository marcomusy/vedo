"""Map an array which is defined on
the vertices of a mesh to its cells"""
from vedo import *

mesh1 = Mesh(dataurl+'icosahedron.vtk').lineWidth(0.1).flat()

doc = Text2D(__doc__, pos="bottom-left")

# let the scalar be the z coordinate of the mesh vertices
msg1 = Text2D("Scalar originally defined on points..", pos="top-center")
mesh1.pointdata["myzscalars"] = mesh1.points()[:, 2]

mesh1.cmap("jet", "myzscalars", on="points")

msg2 = Text2D("..is interpolated to cells.", pos="top-center")
mesh2 = mesh1.clone(deep=False).mapPointsToCells()

show(mesh1, msg1, doc, at=0, N=2, axes=11, viewup="z")
show(mesh2, msg2,      at=1, interactive=True).close()
