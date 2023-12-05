"""Share the same color and transparency
mapping across different volumes"""
from vedo import Volume, show
import numpy as np

arr = np.zeros(shape=(50,60,70))
for i in range(50):
    for j in range(60):
        for k in range(70):
            arr[i,j,k] = k

vol1 = Volume(arr   ).mode(1).cmap('jet', alpha=[0,1], vmin=0, vmax=80).add_scalarbar("vol1")
vol2 = Volume(arr+30).mode(1).cmap('jet', alpha=[0,1], vmin=0, vmax=80).add_scalarbar("vol2")

# or equivalently, to set transparency:
# vol1.alpha([0,1], vmin=0, vmax=70)

show([(vol2, __doc__), vol1], shape=(2,1), axes=1, elevation=-25)
