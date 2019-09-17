from vtkplotter import *
import numpy as np

data1 = np.random.randn(500)*3+10
data2 = np.random.randn(500)*2+ 7

h1 = histogram(data1, fill=True, outline=False, errors=True)
h2 = histogram(data2, fill=False, lc='firebrick', lw=4)
h2.z(0.1) # put h2 in front of h1

h1.scale([1, 0.2, 1]) # set a common y-scaling factor
h2.scale([1, 0.2, 1])

show(h1, h2, bg='white', axes=1)