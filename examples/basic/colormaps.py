"""
Example usage of cmap() to assign a color to each mesh vertex
by looking it up in matplotlib database of colormaps
"""
print(__doc__)
from vedo import Plotter, datadir

# these are the some matplotlib color maps
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

vp = Plotter(N=len(mapkeys))
vp.legendSize = 0.4

mug = vp.load(datadir+"mug.ply")

scalars = mug.points()[:, 1]  # let y-coord be the scalar

for i, key in enumerate(mapkeys):  # for each available color map name
    imug = mug.clone().cmap(key, scalars, n=5)
    vp.show(imug, key, at=i)

vp.show(interactive=1)
