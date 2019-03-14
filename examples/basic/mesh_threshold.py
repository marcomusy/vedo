"""
Extracts the cells where scalar value 
satisfies a threshold criterion.
"""
from vtkplotter import load, Text, show

doc = Text(__doc__)

man = load("data/shapes/man.vtk")

scals = man.coordinates()[:, 1] + 37  # pick y coords of vertices

man.pointColors(scals, cmap="cool")
man.addScalarBar(title="threshold", horizontal=True)

# make a copy and threshold the mesh
cutman = man.clone().threshold(scals, vmin=36.9, vmax=37.5)

# distribute the actors on 2 renderers
show([[man, doc], cutman], N=2, elevation=-30, axes=0)
