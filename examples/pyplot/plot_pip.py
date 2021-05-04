"""Picture in picture plotting"""
from vedo import show
from vedo.pyplot import plot, settings
import numpy as np

settings.defaultFont = 'Theemim'

x = np.arange(0, 4, 0.1)
y1 = 3*np.exp(-x)
y2 = 3*np.exp(-x)*np.cos(2*x)**2

axes_opts = dict(numberOfDivisions=3, xyPlaneColor='lavender', xyAlpha=1)

# Build first plot and its axes:
plt1 = plot(x, y1,
            title=__doc__,
            xtitle='time in seconds',
            ytitle='some function [a.u.]',
)

# Build second plot and its axes:
plt2 = plot(x, y2,
            title='my second plot',
            xtitle='time in seconds',
            ytitle='some other function',
            lc='red',
            pad=0,          # no margins
            axes=axes_opts,
)

# Scale the plot2 to make it small
#  and position it anywhere in the scene:
plt2.scale(0.5).pos(2, 1.4, 0.01)

show(plt1, plt2, zoom=1.1).close()

