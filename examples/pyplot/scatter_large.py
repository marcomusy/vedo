"""Scatter plot of 1M points with
assigned colors and transparencies.

Use mouse to zoom,
press r to reset,
press p to increase point size."""
from vedo import *

N = 1000000

x = np.random.rand(N)
y = np.random.rand(N)
RGBA = np.c_[x*255, y*255, np.zeros(N), y*255]

pts = np.array([x,y]).T
pts = Points(pts).point_size(1)
pts.pointcolors = RGBA

# use mouse to zoom, press r to reset
show(pts, __doc__, axes=1, mode="RubberBandZoom").close()
