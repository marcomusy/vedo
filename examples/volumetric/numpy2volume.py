# Make a Volume from a numpy object
#
import numpy as np
from vtkplotter import *

X, Y, Z = np.mgrid[:30, :30, :30]
# scaled distance from the center at (15, 15, 15)
scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2)/225
print('scalar min, max =', np.min(scalar_field), np.max(scalar_field))

vol = Volume(scalar_field)
lego = vol.legosurface(vmin=1, vmax=2)
text1 = Text('Make a Volume from a numpy object', c='blue')
text2 = Text('lego isosurface representation\nvmin=1, vmax=2', c='darkred')

print('numpy array from Volume:', vol.getPointArray().shape)

show([(vol,text1), (lego,text2)], N=2, bg='white')
