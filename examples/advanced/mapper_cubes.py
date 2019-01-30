'''
Cubes.
Reusing the same actor mapper make visualization faster.
'''
print(__doc__)
from vtkplotter import cube, show, ProgressBar
import numpy as np

N = 10000

centers = np.random.rand(N, 3)

cube0 = cube(length=.01)

cubes=[]

for i,cn in enumerate(centers):
    acube = cube(cn, length=.01, c=i)
    acube.SetMapper(cube0.mapper) # reuse same mapper
    cubes.append(acube)

show(cubes, verbose=0)