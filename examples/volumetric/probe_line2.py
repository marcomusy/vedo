"""Probe a Volume with a line and plot the intensity values"""
from vedo import dataurl, Volume, Line, show
from vedo.pyplot import plot

vol = Volume(dataurl + "embryo.slc")
vol.add_scalarbar3d("wild-type mouse embryo", c="k")
vol.scalarbar = vol.scalarbar.clone2d("bottom-right", 0.2)

p1, p2 = (50, 50, 50), (200, 200, 200)
pl = Line(p1, p2, res=100).lw(4)

# Probe the Volume with the line
pl.probe(vol)

# Get the probed values along the line
xvals = pl.vertices[:, 0]
yvals = pl.pointdata[0]

# Plot the intensity values
fig = plot(
    xvals,
    yvals,
    xtitle=" ",
    ytitle="voxel intensity",
    aspect=16 / 9,
    spline=True,
    lc="r",  # line color
    marker="O",  # marker style
)
fig = fig.shift(0, 25, 0).clone2d()

show(vol, pl, fig, __doc__, axes=14).close()
