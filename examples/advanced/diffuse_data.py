"""
Diffuse sparse point values over a grid and inspect the result distribution.
Then demonstrate Laplacian diffusion of a scalar field on a surface mesh.
"""

import numpy as np
from vedo import Grid, Sphere, settings, show
from vedo.pyplot import histogram

settings.default_font = "FiraMonoMedium"

########################################
# Section 1: smooth_data on a flat Grid
########################################
grid = Grid(res=[50, 50])
grid.wireframe(False).lw(0)

# Two point sources on the grid.
values = np.zeros(grid.npoints)
values[int(grid.npoints / 2)] = 1
values[int(grid.npoints / 5)] = 1

grid.pointdata["scalars"] = values
grid.cmap("Set1_r").add_scalarbar()

grid2 = grid.clone()
# Iterative scalar diffusion.
grid2.smooth_data(niter=750, relaxation_factor=0.1, strategy=1)
grid2.cmap("Set1_r").add_scalarbar()

his = histogram(
    grid2.pointdata["scalars"],
    c="k4",
    xtitle="Concentration",
    ytitle="Frequency",
    axes=dict(htitle="", axes_linewidth=2, xyframe_line=0),
)
his = his.clone2d()  # anchor it to screen coords

print("integrated over domain:", grid2.integrate_data())

show(
    [["Initial state", grid], ["After diffusion", grid2, his]],
    N=2,
    axes=1,
).close()


########################################
# Section 2: laplacian_diffusion on a Mesh
########################################
sph = Sphere(res=100).rotate_x(30)
sph.cut_with_plane(normal=[0, 0, 1], origin=[0, 0, 0.5])

# Sharp Gaussian spike at the apex of the cut mesh.
u = np.exp(-500 * (sph.coordinates[:, 2] - 1.0) ** 2)
sph.pointdata["u"] = u

before = sph.clone().cmap("rainbow", "u", vmin=0, vmax=1)
sph.laplacian_diffusion("u", dt=0.0001, num_steps=100)
after = sph.cmap("rainbow", "u", vmin=0, vmax=1).add_scalarbar()

intg1 = before.integrate_data()["pointdata"]["u"][0]
intg2 = after.integrate_data()["pointdata"]["u"][0]
txt1 = "Before Laplacian diffusion\n∫u dA = {:.6f}".format(intg1)
txt2 = "After 100 steps (dt=0.0001)\n∫u dA = {:.6f}".format(intg2)
show([[txt1, before], [txt2, after]], N=2, axes=1).close()
