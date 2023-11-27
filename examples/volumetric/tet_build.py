"""Build a TetMesh (tetrahedral mesh)
by manually defining vertices and cells"""
from vedo import *

points = [ 
    (0, 0, 0), # first tet
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 2),
    (3, 3, 3), # second tet
    (4, 3, 3),
    (4, 4, 3),
    (3, 4, 4),
    (2, 5, 3), # third tet
    (3, 5, 3),
    (3, 6, 3),
    (2, 5, 5),
]

# Cells are defined by a list of 4 vertex indices
# note that "cells" and "tetrahedrons" are the same thing
tets = [[0,1,2,3], [4,5,6,7], [8,9,10,11]]

# Define a scalar value for each cell we have created
values = np.array([10.0, 20.0, 30.0])

# Create the TeTMesh object and assign any number of data arrays to it
tm = TetMesh([points, tets])
tm.celldata["myscalar1"] = values
tm.celldata["myscalar2"] = -values / 10
tm.pointdata["myvector"] = np.random.rand(tm.npoints)
# ...

print(tm)

tm.celldata.select("myscalar2").cmap('jet').add_scalarbar()
# tm.color('green') # or set a single color

# Create labels for the vertices
labels = tm.labels2d('id', scale=2)

show(tm, labels, __doc__, axes=1).close()
