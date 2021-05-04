"""Extracts cells of a Mesh which satisfy
the threshold criterion: 37 < scalar < 37.5"""
from vedo import *

man = Mesh(dataurl+"man.vtk")

scals = man.points()[:, 0] + 37  # pick y coords of vertices

man.cmap("cool", scals).addScalarBar(title="threshold", horizontal=True)

# make a copy and threshold the mesh
cutman = man.clone().threshold(scals, 37, 37.5)

# distribute the meshes on the 2 renderers
show([(man, __doc__), cutman], N=2, elevation=-30, axes=11).close()
