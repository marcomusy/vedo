"""Colorize a large cloud of 1M points by passing
colors and transparencies in the format (R,G,B,A)"""
from vedo import *
import numpy as np
import time

N = 1000000

pts = np.random.rand(N, 3)
RGB = pts * 255
Alpha = pts[:, 2] * 255
RGBA = np.c_[RGB, Alpha]  # concatenate

t0 = time.time()

# passing c in format (R,G,B,A) is ~50x faster
pts = Points(pts, r=2, c=RGBA)

t1 = time.time()
print("-> elapsed time:", t1-t0, "seconds for N:", N)

show(pts, __doc__, axes=True).close()
