"""
Visualize a Fenics/dolfin mesh.
Select mesh and press X to slice it.
"""
import dolfin
from vedo.dolfin import plot, download

fpath = download("https://vedo.embl.es/examples/data/dolfin_fine.xml")
mesh1 = dolfin.Mesh(fpath)

plot(mesh1)

# show another light-green mesh in a new plotter window,
# show file header too as an additional text comment
mesh2 = dolfin.UnitCubeMesh(8,8,8)

plot(mesh2, text=__doc__, color='lg', new=True)
