"""Interactive streamlines seeded from a single draggable point widget.
Drag the point to move the seed and recompute the streamline."""

from vedo import dataurl, UnstructuredGrid, Plotter
from vedo.addons import PointWidget


def compute_stream(pos):
    sl = domain.compute_streamlines(
        [pos],
        integrator="rk4",
        direction="both",
        max_propagation=100,
        max_steps=5000,
    )
    sl.name = "Streamline"
    sl.lw(3).cmap("coolwarm")
    return sl


def recompute(w, evt):
    plt.remove("Streamline").add(compute_stream(pw.pos))
    plt.render()


domain = UnstructuredGrid(dataurl + "comb_domain.vtu").alpha(0.05).c("white")

pw = PointWidget((5, 0, 30), c="green7", ps=0.4)
pw.add_observer("interaction", recompute)

stream = compute_stream(pw.pos)

plt = Plotter(axes=7, bg="blackboard")
plt.add(domain, pw, stream, __doc__)
plt.show().close()
