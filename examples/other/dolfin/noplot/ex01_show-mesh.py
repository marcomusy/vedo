"""
Fenics/dolfin mesh interoperability.
Select mesh and press X to slice it.
"""
import dolfin
from vtkplotter.dolfin import MeshActor, Text, show, datadir

mesh1 = dolfin.Mesh(datadir + "dolfin_fine.xml")

show(Text("a dolfin 2d mesh", pos=2), interactive=False)

show(mesh1, interactive=True)  # show mesh and hold on

## show another dolfin mesh
mesh2 = dolfin.UnitCubeMesh(8,8,8)

actor = MeshActor(mesh2)  # build a vtkActor from Mesh

# show Actor and the header comment of this file
show(actor, Text(__doc__), interactive=True)
