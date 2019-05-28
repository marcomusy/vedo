from vtkplotter import *
import numpy as np

data_matrix = np.zeros([75, 75, 75])
# data_matrix[0:35, 0:35, 0:35] = 50
# data_matrix[25:55, 25:55, 25:55] = 100
# data_matrix[45:74, 45:74, 45:74] = 150
# or
for ix in range(75):
    for iy in range(75):
        for iz in range(75):
            data_matrix[ix, iy, iz] = ix + iy + iz

v = Volume(data_matrix)

s = isosurface(v, threshold=[t for t in arange(0, 200, 10)])
s.alpha(0.5).lw(0.1)

show(v, s, N=2, axes=8, bg="w", depthpeeling=1)
