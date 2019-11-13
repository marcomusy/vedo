import numpy as np

data_matrix = np.zeros([75, 75, 75], dtype=np.uint8)
data_matrix[0:35, 0:35, 0:35] = 50
data_matrix[25:55, 25:55, 25:55] = 100
data_matrix[45:74, 45:74, 45:74] = 150

from vtkplotter import Volume

Volume(data_matrix).show(bg="w")
