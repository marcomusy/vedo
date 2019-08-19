"""How to transform/map an array
which is defined on the vertices of a mesh
to its cells with mapPointsToCells()
"""
from vtkplotter import *

mesh1 = load(datadir+'icosahedron.vtk')

# let the scalar be the z coordinate of the mesh vertices
scals = mesh1.getPoints()[:, 2]

mesh1.lineWidth(0.1).addPointScalars(scals, name='scals')
mesh2 = mesh1.clone().addScalarBar(c='w').mapPointsToCells()

doc = Text(__doc__, pos=8, c="w")
msg1 = Text("Scalar originally defined on points..", pos=5, c="w")
msg2 = Text("..is interpolated to cells.", pos=5, c="w")

printInfo(mesh1)
printInfo(mesh2)

show(mesh1, msg1, doc, at=0, N=2, axes=1, viewup="z")
show(mesh2, msg2,      at=1, interactive=True)

