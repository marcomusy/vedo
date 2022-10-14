
from vedo import Arc, vtk_version
import numpy as np

print('-----------------------------------------------------')
print('VTK Version', vtk_version, "test_shapes.py")
print('-----------------------------------------------------')

#####################################
arc = Arc(center=None, point1=(1, 1, 1), point2=None, normal=(0, 0, 1), angle=np.pi)
assert isinstance(arc, Arc)
