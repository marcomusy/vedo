"""Share the same color and trasparency mapping across different volumes"""
from vedo import Volume, Line, show
import numpy as np

arr = np.zeros(shape=(50,50,50))
for i in range(50):
    for j in range(50):
        for k in range(50):
            arr[i,j,k] = j

vol1 = Volume(arr   ).mode(1).cmap('jet', alpha=[0,1], vmin=0, vmax=80).addScalarBar("vol1")
vol2 = Volume(arr+30).mode(1).cmap('jet', alpha=[0,1], vmin=0, vmax=80).addScalarBar("vol2")

# or equivalently, to set transparency:
# vol1.alpha([0,1], vmin=0, vmax=70)

# can also manually build a scalarbar object to span the whole range:
sb = Line([50,0,0],[50,50,0]).cmap('jet',[0,70]).addScalarBar3D("vol2", c='black').scalarbar

show([(vol1, __doc__), (vol2, sb)], N=2, axes=1)
