"""Build a TetMesh (tetrahedral mesh)
by manually defining vertices and cells"""
from vedo import *

points = [ (0, 0, 0), # first tet
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

tets = [[0,1,2,3], [4,5,6,7], [8,9,10,11]]

scal = [10.0, 20.0, 30.0] # cell scalars

# Create the TeTMesh object
tm = TetMesh([points,tets])
tm.celldata["myscal"] = scal

tm.color('jet')
# tm.color('green') # or set a single color

printc("tetmesh.inputdata():", type(tm.inputdata())) # vtkUnstructuredGrid
printc("points, cells      :", len(tm.points()), len(tm.cells()))

# Optionally convert tm to a Mesh (for visualization)
show([(tm, __doc__),
      (tm.tomesh(),"TetMesh.tomesh()"),
     ],  N=2, axes=1,
).close()
