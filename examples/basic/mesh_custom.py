"""Controlling the color and transparency
of a Mesh with various color map definitions
"""
from vedo import *

#useDepthPeeling may improve the rendering of transparent objects
settings.useDepthPeeling = True

man = load(datadir+"man.vtk")

# let the scalar be the z coordinate of the mesh vertices
scals = man.points()[:, 2]

# custom color map with specified opacities
mymap = ["darkblue", "cyan", (1, 0, 0)]
alphas = [0.8, 0.4, 0.2]

# - OR by predefined color numbers:
#mymap = [i for i in range(10)]
#alphas = [i/10. for i in range(10)]

# - OR by generating a palette betwwen 2 colors:
#mymap = makePalette('pink', 'green', N=500, hsv=True)
#alphas = 1

man.cmap(mymap, scals, alpha=alphas).addScalarBar()

show(man, __doc__, viewup="z", axes=7)
