"""Scatter plot of 1M points with
assigned colors and transparencies.

Use mouse to zoom,
press r to reset,
press p to increase point size.
"""
from vedo import *
import numpy as np
import time

N = 1000000

x = np.random.rand(N)
y = np.random.rand(N)
RGBA = np.c_[x*255, y*255, np.zeros(N), y*255]

t0 = time.time()

pts = Points([x,y], r=1, c=RGBA)

t1 = time.time()
print("-> elapsed time:", t1-t0, "seconds for N:", N)

# use mouse to zoom, press r to reset
show(pts, __doc__, axes=1, mode="RubberBandZoom").close()
