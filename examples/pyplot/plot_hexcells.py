"""3D Bar plot of a TOF camera with hexagonal pixels"""
from vedo import *
import numpy as np

settings.defaultFont = "Glasgo"
settings.useParallelProjection = True

vals = np.abs(np.random.randn(4*6))  # pixel heights
cols = colorMap(vals, "summer")

k = 0
items = [__doc__]
for i in range(4):
    for j in range(6):
        val, col= vals[k], cols[k]
        x, y, z = [i+j%2/2, j/1.155, val+0.01]
        zbar= Polygon([x,y,0], nsides=6, r=0.55, c=col).extrude(val)
        line= Polygon([x,y,z], nsides=6, r=0.55, c='k').wireframe().lw(2)
        txt = Text3D(f"{i}/{j}", [x,y,z], s=.15, c='k', justify='center')
        items += [zbar, line, txt]
        k += 1

show(items, axes=7)
