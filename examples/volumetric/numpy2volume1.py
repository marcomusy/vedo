'Make a Volume from a numpy.mgrid'
import numpy as np
from vedo import *

X, Y, Z = np.mgrid[:30, :30, :30]
# scaled distance from the center at (15, 15, 15)
scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2)/225

vol = Volume(scalar_field)
vol.addScalarBar3D()

lego = vol.legosurface(vmin=1, vmax=2)
lego.addScalarBar3D()
text1 = Text2D(__doc__, c='blue')
text2 = Text2D('its lego isosurface representation\nvmin=1, vmax=2', c='dr')

print('numpy array from Volume:', vol.tonumpy().shape)

show([(vol,text1), (lego,text2)], N=2, azimuth=10).close()
