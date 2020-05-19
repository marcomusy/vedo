"""Extracts cells of a Mesh which satisfy
a threshold criterion:
37 < scalar < 37.5
"""
from vtkplotter import *

man = load(datadir+"man.vtk")

scals = man.points()[:, 0] + 37  # pick y coords of vertices

man.pointColors(scals, cmap="cool")
man.addScalarBar(title="threshold", horizontal=True)

# make a copy and threshold the mesh
cutman = man.clone().threshold(scals, 37, 37.5)

# distribute the meshes on 2 renderers
show([(man, __doc__), cutman], N=2, elevation=-30, axes=0)
