"""Diffuse sparse point values over a grid and inspect the result distribution."""
import numpy as np
from vedo import Grid, settings, show
from vedo.pyplot import histogram

settings.default_font = "FiraMonoMedium"

grid = Grid(res=[50,50])
grid.wireframe(False).lw(0)

# Two point sources on the grid.
values = np.zeros(grid.npoints)
values[int(grid.npoints/2)] = 1
values[int(grid.npoints/5)] = 1

grid.pointdata["scalars"] = values
grid.cmap("Set1_r").add_scalarbar()

grid2 = grid.clone()
# Iterative scalar diffusion.
grid2.smooth_data(niter=750, relaxation_factor=0.1, strategy=1)
grid2.cmap("Set1_r").add_scalarbar()

his = histogram(
    grid2.pointdata["scalars"],
    c='k4',
    xtitle="Concentration",
    ytitle="Frequency",
    axes=dict(htitle="", axes_linewidth=2, xyframe_line=0),
)
his = his.clone2d() # anchor it to screen coords

print("integrated over domain:", grid2.integrate_data())

show([
    ["Initial state", grid],
    ["After diffusion", grid2, his]],
    N=2, axes=1,
).close()
