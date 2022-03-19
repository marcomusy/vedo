"""Plotting functions with error bands"""
import numpy as np
from vedo import Rectangle, Text3D, Marker, Line
from vedo.pyplot import plot

# Make up same dummy data
x = np.arange(0, 6, 0.1)
y = 2+2*np.sin(2*x)/(x+1)
ye= y**2 / 10
miny = np.min(y-ye)
idx = np.argmax(y)

# Plot the two variables, return a Plot(Assembly) object:
pl = plot(
    x,y,
    yerrors=ye,
    xtitle='time in \museconds',
    ytitle='y oscillation [a.u.]',
    ylim=(0.5, 5),
    aspect=5/3,      # aspect ratio (any float = x_size/y_size)
    errorBand=True,  # join errors on y into an error band
    lc="red2",       # line color
    ec="red7",       # error band color
    pad=0.0,         # tight margins, no padding
    axes=dict(xyGrid=False, axesLineWidth=2, xyFrameLine=2),
)

# Add a grey transparent rectangle to represent an exclusion region:
pl += Rectangle([1,0.5], [2.7,5], c='grey5').lighting('off')

# Add some text (set z=2 so it stays on top):
pl += Text3D("Excluded\ntime range!", s=.2, c='k', font="Quikhand").rotateZ(20).pos(1.3,3.6,1)

# Add a star marker at maximum of function (set z=0.1, so it stays on top):
pl += Marker('*', pos=(x[idx], y[idx], 0.1), c='blue')

# Add a dashed line to indicate the minimum
pl += Line((x[0], miny), (x[-1], miny)).pattern('- . -').lw(3)

pl.show(zoom='tight', mode="image", size=(900,600)).close()
