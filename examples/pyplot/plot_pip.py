"""Picture in picture plotting"""
from vedo import settings, show
from vedo.pyplot import plot
import numpy as np

settings.defaultFont = 'Theemim'
settings.useParallelProjection = True

x = np.arange(0, 4, 0.01)
y1 = 3*np.exp(-x)
y2 = 3*np.exp(-x)*np.cos(2*x)**2


# Build first plot and its axes:
fig1 = plot(x, y1,
            title=__doc__,
            xtitle='time in seconds',
            ytitle='some function [a.u.]',
)

# Build second plot and its axes:
axes_opts = dict(numberOfDivisions=3,
                 gridLineWidth=0,
                 xyPlaneColor='lightblue',
                 xyAlpha=1,
                 textScale=1.8,
)
fig2 = plot(x, y2,
            xtitle=' ',
            ytitle='some other function',
            lc='red5',
            pad=0,          # no margins
            axes=axes_opts,
)

# Scale the plot2 to make it small
#  and position it anywhere in the scene:
fig2.scale(0.5).pos(2, 1.4, 0.1)

show(fig1, fig2, zoom='tight', mode='image').close()

