"""
Example on how to specify a color for each
individual cell or point of an actor's mesh.
Keyword depthpeeling may improve the rendering of transparent objects.
"""
from vtkplotter import *

doc = Text(__doc__, pos=1, c="w")


man = load(datadir+"man.vtk")

# let the scalar be the z coordinate of the mesh vertices
scals = man.coordinates()[:, 2]

# custom color map with specified opacities
#mymap = ["darkblue", "cyan", (1, 0, 0)]
#alphas = [0.8, 0.4, 0.2]

# - OR by predefined color numbers:
mymap = [i for i in range(10)]
alphas = [i/10. for i in range(10)]

# - OR by generating a palette betwwen 2 colors:
# from vtkplotter.colors import makePalette
#mymap = makePalette('pink', 'green', N=500, hsv=True)
#alphas = 1

man.pointColors(scals, cmap=mymap, alpha=alphas).addScalarBar()

show(man, doc, viewup="z", axes=8, depthpeeling=1)
