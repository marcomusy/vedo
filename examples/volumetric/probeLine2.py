"""Probe a Volume with a line
and plot the intensity values"""
from vedo import *
from vedo.pyplot import plot

vol = load(datadir+'vase.vti')
vol.addScalarBar3D(title='vase', c='k', italic=1)

p1, p2 = (10,10,10), (90,90,90)
pl = probeLine(vol, p1, p2, res=50).lineWidth(4)

xvals = pl.points()[:,0]
yvals = pl.getPointArray()

plt = plot(xvals, yvals,
           spline=True,
           lc="r",       # line color
           marker="*",   # marker style
           mc="dr",      # marker color
           ms=0.6,       # marker size
          )

show([(vol, pl, __doc__), plt], N=2, sharecam=False)
