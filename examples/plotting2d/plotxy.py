from vtkplotter import plotxy, show
import numpy as np

x  = np.arange(0, 10, 1)
y  = np.sin(x)

# assign errors to both x and y
ye = np.random.rand(10)/2
xe = np.random.rand(10)

##############
plt1 = plotxy(
    [x, y],       # accepts different formats
    yscale=3,     # set an optional y-scaling factor
    xlimits=(-1, 11),
    splined=False,
    lc="r",       # line color
    marker="*",   # marker style
    mc="dr",      # marker color
)

##############
plt2 = plotxy(
    [x+1, y+0.2],
    xerrors=xe,   # show error bars
    yerrors=ye,
    yscale=3,     # choose the same y-scale as above!
    splined=True,
    xtitle="x variable (mm)",
    ytitle="y(x)",
    lc="b",
    marker="s",   # o, p, *, h, D, d , v, ^, s, x, a
)

##############
show(plt1, plt2, bg="w", axes=1)
