"""Histogram of 2 variables as 3D bars"""
import numpy as np
from vedo import Points, show
from vedo.pyplot import histogram

n = 1000
x = np.random.randn(n)*1.5 + 60
y = np.random.randn(n)     + 70

histo = histogram(
    x, y,
    bins=(10, 15),
    cmap="summer",
    ztitle="Number of entries in bin",
    mode="3d",
    gap=0.0,
    zscale=0.4,  # rescale the z axis
    aspect=2
)

print(histo.frequencies)

# Add also the original points on top
histo += Points(np.c_[x, y], r=4, c="red5").z(3)

show(histo, __doc__, elevation=-80).close()
