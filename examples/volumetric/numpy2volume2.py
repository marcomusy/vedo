"""Create a Volume from a numpy array"""
import numpy as np
from vedo import Volume, show

data_matrix = np.zeros([70, 80, 90], dtype=np.uint8)
data_matrix[ 0:30,  0:30,  0:30] = 1
data_matrix[30:50, 30:60, 30:70] = 2
data_matrix[50:70, 60:80, 70:90] = 3

vol = Volume(data_matrix)
vol.cmap(['white','b','g','r']).mode(1)
vol.add_scalarbar()

show(vol, __doc__, axes=1).close()
