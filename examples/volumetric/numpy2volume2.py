"""Create a Volume from a numpy array"""
import numpy as np

data_matrix = np.zeros([75, 75, 75], dtype=np.uint8)
# all voxels have value zero except:
data_matrix[0:35,   0:35,  0:35] = 1
data_matrix[35:55, 35:55, 35:55] = 2
data_matrix[55:74, 55:74, 55:74] = 3

from vedo import Volume, show

vol = Volume(data_matrix, c=['white','b','g','r'], mapper='gpu')
vol.addScalarBar3D()

# optionally mask some parts of the volume (needs mapper='gpu'):
# data_mask = np.zeros_like(data_matrix)
# data_mask[10:65, 10:45, 20:75] = 1
# vol.mask(data_mask)

show(vol, __doc__, axes=1).close()
