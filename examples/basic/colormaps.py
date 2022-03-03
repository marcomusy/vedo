"""
Example usage of cmap() to assign a color to each mesh vertex
by looking it up in matplotlib database of colormaps
"""
print(__doc__)
from vedo import Plotter, Mesh, dataurl

# these are the some matplotlib color maps
maps = [
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

mug = Mesh(dataurl+"mug.ply")
scalars = mug.points()[:, 1]  # let y-coord be the scalar

plt = Plotter(N=len(maps))

for i, key in enumerate(maps):  # for each available color map name
    imug = mug.clone(deep=False).cmap(key, scalars, n=5)
    plt.at(i).show(imug, key)

plt.interactive().close()
