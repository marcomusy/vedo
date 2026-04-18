"""Interactive streamlines seeded along a draggable line widget.
Drag the line handles to move the seed line."""

import numpy as np
from vedo import dataurl, UnstructuredGrid, Plotter
from vedo.addons import LineWidget


def compute_streams(seeds):
    sl = domain.compute_streamlines(
        seeds,
        integrator="rk4",
        direction="forward",
        max_propagation=50,
        max_steps=5000,
    )
    sl.name = "Streamlines"
    sl.lw(2).cmap("coolwarm")
    return sl


def recompute(w, evt):
    new_streams = compute_streams(lw.points)
    plt.remove("Streamlines").add(new_streams)


domain = UnstructuredGrid(dataurl + "comb_domain.vtu")
domain.alpha(0.05).c("white")

lw = LineWidget(
    [5.0, -5.0, 30.0], [5.0, 5.0, 30.0],
    lc="yellow", pc="yellow", lw=4, ps=12, res=32
)
lw.add_observer("interaction", recompute)
streams = compute_streams(lw.points)

plt = Plotter(axes=7, bg="blackboard")
plt.add(domain, lw, streams, __doc__)
plt.show().close()
