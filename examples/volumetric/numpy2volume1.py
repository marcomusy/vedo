"""Create a Volume from a numpy.mgrid"""
import numpy as np
from vedo import Volume, Text2D, show

X, Y, Z = np.mgrid[:30, :30, :30]
# Distance from the center at (15, 15, 15)
scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2) /225

vol = Volume(scalar_field)
vol.addScalarBar3D()
print('numpy array from Volume:', vol.tonumpy().shape)

lego = vol.legosurface(vmin=1, vmax=2)
lego.cmap('hot_r', vmin=1, vmax=2).addScalarBar3D()

text1 = Text2D(__doc__, c='blue')
text2 = Text2D('..and its lego isosurface representation\nvmin=1, vmax=2', c='dr')

show([(vol,text1), (lego,text2)], N=2, azimuth=10).close()
