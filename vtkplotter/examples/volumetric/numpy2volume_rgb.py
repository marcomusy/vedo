"""Modify a Volume dataset and
colorize voxels individually
"""
from vtkplotter import Volume, show
import numpy as np

vol = Volume(dims=(10,11,12), mode=0)
vol.alpha(0.8).jittering(False)
vol.interpolation(0) # nearest neighbour interpolation type

# Overwrite the (sofar empty) voxel data with new data
vol.imagedata().AllocateScalars(3, 4) # type 3 corresponds to np.uint8
arr = vol.getPointArray()
arr[:] = np.random.randint(0,255, (10*11*12,4)).astype(np.uint8)

# For 4 component data, the first 3 directly represent RGB, no lookup table.
# (see https://vtk.org/doc/nightly/html/classvtkVolumeProperty.html):
vol.GetProperty().IndependentComponentsOff()

show(vol, __doc__, axes=1)