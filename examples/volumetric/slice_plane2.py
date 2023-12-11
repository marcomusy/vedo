"""Slice a Volume with multiple planes
Make low values of the scalar completely transparent"""
from vedo import *

vol = Volume(dataurl+'embryo.slc')
vol.cmap('bone').alpha([0,0,0.5])

slices = []
for i in range(4):
    sl = vol.slice_plane(origin=[150,150,i*50+50], normal=(0,-1,1))
    slices.append(sl)

amap = [0, 1, 1, 1, 1]  # hide low value points giving them alpha 0
mslices = merge(slices) # merge all slices into a single Mesh
mslices.cmap('hot_r', alpha=amap).lighting('off')

# Add a 3d scalarbar to the scene
mslices.add_scalarbar3d()
# make 2d copy of it and move it to the center-right
mslices.scalarbar = mslices.scalarbar.clone2d("center-right", 0.15)

show(vol, mslices, __doc__, axes=1).close()
