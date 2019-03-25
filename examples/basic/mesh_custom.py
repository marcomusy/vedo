"""
Example on how to specify a color for each
individual cell or point of an actor's mesh.
Keyword depthpeeling may improve the rendering of transparent objects.
"""
from vtkplotter import load, Text, show, datadir

doc = Text(__doc__, pos=1, c="w")


man = load(datadir+"shapes/man.vtk")

# let the scalar be the z coordinate of the mesh vertices
scals = man.coordinates()[:, 2]

# custom color map with optional opacity (here: 1, 0.2 and 0.8)
mymap = ["darkblue", "cyan 0.2", (1, 0, 0, 0.8)]

# - or by predefined color numbers:
# mymap = [i for i in range(10)]

# - or by generating a palette betwwen 2 colors:
# from vtkplotter.colors import makePalette
# mymap = makePalette('pink', 'green', N=500, hsv=True)

man.pointColors(scals, cmap=mymap).addScalarBar()

show([man, doc], viewup="z", axes=8, depthpeeling=1)
