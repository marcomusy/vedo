"""Probe a Volume with a line
and plot the probed values"""
from vtkplotter import *

comment = Text2D(__doc__)

vol = load(datadir+'vase.vti')

p1, p2 = (10,10,10), (90,90,90)
pl = probeLine(vol, p1, p2, res=50).lw(3)

xvals = pl.points()[:,0]
yvals = pl.getPointArray()

plt = plot(xvals, yvals,
           spline=True,
           lc="r",       # line color
           marker="*",   # marker style
           mc="dr",      # marker color
           ms=0.6,       # marker size
          )

#show(vol, pl, comment, plt, axes=1, bg='w')
show([(vol, pl, comment), plt], N=2, sharecam=0, axes=1)
