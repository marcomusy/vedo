"""
Example usage of pointColors to assign a color to each mesh vertex 
by looking it up in matplotlib database of colormaps
"""
print(__doc__)
from vtkplotter import Plotter

# these are the available color maps
mapkeys = [
    "afmhot",
    "binary",
    "bone",
    "cool",
    "coolwarm",
    "copper",
    "gist_earth",
    "gray",
    "hot",
    "jet",
    "rainbow",
    "winter",
]

vp = Plotter(N=len(mapkeys), axes=4)
vp.legendSize = 0.4

mug = vp.load("data/shapes/mug.ply")
scalars = mug.coordinates()[:, 1]  # let y-coord be the scalar

for i, key in enumerate(mapkeys):  # for each available color map name
    imug = mug.clone().legend(key)
    imug.pointColors(scalars, cmap=key)
    vp.show(imug, at=i)

vp.show(interactive=1)
