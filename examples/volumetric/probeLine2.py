"""Probe a Volume with a line
and plot the intensity values"""
from vedo import dataurl, Volume, probe_line, show
from vedo.pyplot import plot

vol = Volume(dataurl+'embryo.slc')
vol.add_scalarbar_3d('wild-type mouse embryo', c='k')

p1, p2 = (50,50,50), (200,200,200)
pl = probe_line(vol, p1, p2, res=100).linewidth(4)

xvals = pl.points()[:,0]
yvals = pl.pointdata[0] # get the probed values along the line

fig = plot(
    xvals, yvals,
    xtitle=" ", ytitle="voxel intensity",
    aspect=16/9,
    spline=True,
    lc="r",       # line color
    marker="*",   # marker style
    mc="dr",      # marker color
    ms=0.9,       # marker size
)
fig.shift(0,25,0)

show(vol, pl, fig, __doc__, axes=dict(xygrid=0, yzgrid=0)).close()
