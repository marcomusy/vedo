"""Controlling the color and transparency
of a Mesh with various color map definitions"""
from vedo import *

man = Mesh(dataurl + "man.vtk")

# let the scalar be the z coordinate of the mesh vertices
scals = man.vertices[:, 2]

# assign color map with specified opacities
try:
    import colorcet  # https://colorcet.holoviz.org
    mycmap = colorcet.bmy
    alphas = np.linspace(0.8, 0.2, num=len(mycmap))
except ModuleNotFoundError:
    printc("colorcet is not available, use custom cmap", c='y')
    mycmap = ["darkblue", "magenta", (1, 1, 0)]
    alphas = [0.8,              0.6,       0.2]

# - OR by generating a palette between 2 colors:
#mycmap = makePalette('pink', 'green', N=500, hsv=True)
#alphas = 1

man.cmap(mycmap, scals, alpha=alphas).add_scalarbar()

show(man, __doc__, viewup="z", axes=7).close()

