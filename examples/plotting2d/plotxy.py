from vtkplotter import plotxy, show
import numpy as np

x = np.arange(0, 10, 1)
y = np.sin(x)

plt1 = plotxy(
    [x, y],
    yscale=3, # set an optional scaling factor
    xlimits=(-1, 11),
    splined=False,
    lc="r",
    marker="*",
    mc="dr",
)

plt2 = plotxy(
    [x+1, y+0.2],
    yscale=3, # choose the same y-scale as above
    splined=True,
    xtitle="x variable (mm)",
    ytitle="y(x)",
    lc="b",
    marker="D",
)

show(plt1, plt2, bg="w", axes=1)
