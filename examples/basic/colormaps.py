"""
Example usage of cmap() to assign a color to each mesh vertex
by looking it up in matplotlib database of colormaps
"""
from vedo import Plotter, Mesh, dataurl
print(__doc__)

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
scalars = mug.vertices[:, 1]  # let y-coord be the scalar

plt = Plotter(N=len(maps))

for i, key in enumerate(maps):  # for each available color map name
    imug = mug.clone(deep=False).cmap(key, scalars, n_colors=5)
    plt.at(i).show(imug, key)

plt.interactive().close()
