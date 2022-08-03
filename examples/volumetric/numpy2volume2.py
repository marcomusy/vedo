"""Create a Volume from a numpy array"""
import numpy as np
from vedo import Volume, show

data_matrix = np.zeros([70, 80, 90], dtype=np.uint8)
data_matrix[0:30,   0:30,  0:30] = 1
data_matrix[30:50, 30:60, 30:70] = 2
data_matrix[50:69, 60:79, 70:89] = 3

vol = Volume(data_matrix, c=['white','b','g','r'], mapper='gpu')
vol.addScalarBar3D()

# optionally mask some parts of the volume (needs mapper='gpu'):
# data_mask = np.zeros_like(data_matrix)
# data_mask[10:65, 10:65, 20:75] = 1
# vol.mask(data_mask)

show(vol, __doc__, axes=1).close()
