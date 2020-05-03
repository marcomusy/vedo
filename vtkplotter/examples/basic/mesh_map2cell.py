"""Map an array initially
defined on the vertices of a mesh
to its cells with mapPointsToCells()
"""
from vtkplotter import *

mesh1 = load(datadir+'icosahedron.vtk')
mesh1.lineWidth(0.1)

doc = Text2D(__doc__, pos=8)

# let the scalar be the z coordinate of the mesh vertices
msg1 = Text2D("Scalar originally defined on points..", pos=5)
scals = mesh1.points()[:, 2]
mesh1.addPointArray(scals, name='scals')

msg2 = Text2D("..is interpolated to cells.", pos=5)
mesh2 = mesh1.clone().mapPointsToCells()

show(mesh1, msg1, doc, at=0, N=2, axes=1, viewup="z")
show(mesh2, msg2,      at=1, interactive=True)
