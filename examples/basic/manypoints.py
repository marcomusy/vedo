"""
Colorize a large cloud of points by passing
colors and transparencies in the format (R,G,B,A)
"""
print(__doc__)
from vtkplotter import Points
import numpy as np
import time

N = 1000000

pts = np.random.rand(N, 3)
RGB = pts * 255
Alpha = pts[:, 2] * 255
RGBA = np.c_[RGB, Alpha]  # concatenate

print("clock starts")
t0 = time.clock()

# passing c in format (R,G,B,A) is ~50x faster
pts = Points(pts, r=2, c=RGBA) #fast
#pts = Points(pts, r=2, c=pts, alpha=pts[:, 2]) #slow

t1 = time.clock()
print("----> elapsed time:", t1-t0, "seconds for N:", N)

pts.show(bg="white", axes=True)
