"""Hover mouse onto an object
to pop a flag-style label"""
from vedo import *

b = Mesh(dataurl + "bunny.obj")
b.color("purple5").legend("Bugs the bunny")
c = Cube(side=0.1).y(-0.02).compute_normals()
c.alpha(0.8).lighting("off").lw(1).legend("The Cube box")

cap = c.caption(
    "2d caption for a cube\nwith face indices",
    point=[0.044, 0.03, -0.04],
    size=(0.3, 0.06),
    font="VictorMono",
    alpha=1,
)

# create a new object made of polygonal text labels to indicate the cell numbers
flabs = c.labels("id", on="cells", font="Theemim", scale=0.02, c="k")
vlabs = c.clone().clean().labels2d(font="ComicMono", scale=3, bc="orange7")

# create a custom entry to the legend
lbox = LegendBox([b, c], font="Bongas", width=0.25, bg='blue6')

with Plotter(axes=11, bg2="linen") as plt:
    plt.add(b, c, cap, flabs, vlabs, lbox, __doc__)
    plt.add_hint(b, "My fave bunny")
    plt.show()
