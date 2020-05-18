# Scatter plot of a gaussian distribution
# with varying color and point sizes
from vtkplotter import *
from vtkplotter.pyplot import plot
import numpy as np

n = 1000
x = np.random.randn(n)
y = np.random.randn(n)

# define what size must have each marker:
marker_sizes = np.sin(2*x)/8

# define a (r,g,b) list of colors for each marker:
marker_cols = np.c_[np.cos(2*x), np.zeros(n), np.zeros(n)]


txt0 = Text2D("A scatter plot of a\n 2D gaussian distribution")
plt0 = plot(x, y, ma=0.3, lw=0,        # ma = marker alpha
            marker="*",                # marker style
            xtitle="variable A",
            ytitle="variable B",
           )

txt1 = Text2D(" marker size = sin(2x) ")
plt1 = plot(x, y, ma=0.3, lw=0,
            marker="*",                # marker style
            ms=marker_sizes,           # VARIABLE marker sizes
            mc='red',                  # same color for markers
           )

txt2 = Text2D(" marker size = sin(2x)\n red level   = cos(2x)")
plt2 = plot(x, y, ma=0.3, lw=0,
            marker=">",                # marker style
            ms=marker_sizes,           # VARIABLE marker sizes
            mc=marker_cols,            # VARIABLE marker colors
           )

show(plt0, txt0, at=0, N=3, axes=0)
show(plt1, txt1, at=1)
show(plt2, txt2, at=2, interactive=True)
