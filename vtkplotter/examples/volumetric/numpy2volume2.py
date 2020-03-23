import numpy as np

data_matrix = np.zeros([75, 75, 75], dtype=np.uint8)
# all voxels have value zero except:
data_matrix[0:35,   0:35,  0:35] = 1
data_matrix[35:55, 35:55, 35:55] = 2
data_matrix[55:74, 55:74, 55:74] = 3

from vtkplotter import Volume

vol = Volume(data_matrix, c=['white','b','g','r'])
vol.show(axes=1)
