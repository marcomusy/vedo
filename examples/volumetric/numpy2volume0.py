"""Modify a Volume in-place from a numpy array"""
from vedo import Volume, dataurl, show

vol = Volume(dataurl+"embryo.tif")

# `tonumpy()` exposes the voxel buffer for in-place edits.
arr = vol.tonumpy()
arr[:] = arr/5 + 15  # modify the array in-place with [:]

vol.modified()

show(vol, __doc__, axes=1).close()
