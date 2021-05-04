"""A simple scatter plot"""
from vedo import show
from vedo.pyplot import plot
import numpy as np

x = np.random.randn(100)+10
y = np.random.randn(100)*20

plt = plot( x, y,         
            lw=0,
            xtitle="variable x",
            ytitle="variable y",
            aspect=4/3,   # aspect ratio
            marker="*",   # marker style
            mc="dr",      # marker color
            axes=True,
)

# show Assembly object and lock interaction to 2d:
# (can zoom in a region w/ mouse, press r to reset)
show(plt, __doc__, zoom=1.2, viewup='2d').close()
