"""Interactive streamlines seeded from a sphere widget.
Drag the sphere to move the seed cloud, 
resize it by right-click and dragging."""

from vedo import dataurl, UnstructuredGrid, Plotter
from vedo.addons import SphereWidget


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
    plt.remove("Streamlines").add(compute_streams(sw.points))


domain = UnstructuredGrid(dataurl + "comb_domain.vtu").alpha(0.05).c("white")

sw = SphereWidget(center=(5, 0, 30), r=2, c="green7", alpha=0.3, res=6)
sw.add_observer("interaction", recompute)

streams = compute_streams(sw.points)

plt = Plotter(axes=7, bg="blackboard")
plt.add(domain, sw, streams, __doc__)
plt.show().close()
