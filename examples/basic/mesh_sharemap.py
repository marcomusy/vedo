"""Share the same color map
across different meshes"""
from vedo import Mesh, show, dataurl


#####################################
man1 = Mesh(dataurl+"man.vtk")
scals = man1.points()[:, 2] * 5 + 27  # pick z coordinates [18->34]
man1.cmap("rainbow", scals, vmin=18, vmax=44)

#####################################
man2 = Mesh(dataurl+"man.vtk")
scals = man2.points()[:, 2] * 5 + 37  # pick z coordinates [28->44]
man2.cmap("rainbow", scals, vmin=18, vmax=44).addScalarBar()

show([(man1, __doc__), man2], N=2, elevation=-40, axes=11).close()
