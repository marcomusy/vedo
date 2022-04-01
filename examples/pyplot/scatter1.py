"""A simple scatter plot"""
from vedo import show
from vedo.pyplot import plot
import numpy as np

x = np.random.randn(100) + 10
y = np.random.randn(100) * 20 + 20

fig = plot(
    x, y,
    lw=0,         # do not join points with lines
    xtitle="variable x",
    ytitle="variable y",
    marker="*",   # marker style
    mc="dr",      # marker color
)

show(fig, __doc__, zoom='tight').close()
