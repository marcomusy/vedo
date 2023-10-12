"""Extracts cells of a Mesh which satisfy
the threshold criterion: 37 < scalar < 37.5"""
from vedo import *

man = Mesh(dataurl+"man.vtk")

scals = man.vertices[:, 0] + 37  # pick y coords of vertices

# scals data is added to mesh points with automatic name PointScalars
man.cmap("cool", scals).add_scalarbar(title="threshold", horizontal=True)

# make a copy and threshold the mesh
cutman = man.clone().threshold("Scalars", 37, 37.5)

# distribute the meshes on the 2 renderers
show([(man, __doc__), cutman], N=2, elevation=-30, axes=11).close()
