
from vtkplotter import Arc
import numpy as np
import vtk

print('---------------------------------')
print('vtkVersion', vtk.vtkVersion().GetVTKVersion())
print('---------------------------------')

#####################################
arc = Arc(center=None, point1=(1, 1, 1), point2=None, normal=(0, 0, 1), angle=np.pi)
assert isinstance(arc, Arc)
