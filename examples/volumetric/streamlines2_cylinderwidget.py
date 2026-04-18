"""Interactive streamlines seeded from a cylinder widget.
Drag the cylinder to move and resize the seed surface."""

from vedo import dataurl, UnstructuredGrid, Points, Plotter
from vedo.addons import CylinderWidget


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
    plt.remove("Streamlines").add(compute_streams(Points(cw.points)))
    plt.render()


domain = UnstructuredGrid(dataurl + "comb_domain.vtu").alpha(0.05).c("white")

cw = CylinderWidget((5, 0, 30), domain, r=2, axis=(0, 1, 0), c="green5", alpha=0.3, res=(12, 8))
cw.add_observer("interaction", recompute)

streams = compute_streams(Points(cw.points))

plt = Plotter(axes=7, bg="blackboard")
plt.add(domain, cw, streams, __doc__)
plt.show().close()
