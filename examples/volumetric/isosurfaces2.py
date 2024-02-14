"""Isosurface extraction from a volume dataset with discrete values."""
from vedo import *

lut = build_lut(
    [
        [0, "lightyellow"],
        [1, "red"],
        [2, "blue"],
        [3, "yellow"],
        [4, "orange"],
        [5, "cyan"],
        [6, "magenta"],
        [7, "white"],
        [8, "pink"],
        [9, "brown"],
        [10, "lightblue"],
        [11, "lightgreen"],
    ],
    interpolate=False
)

# this dataset is a 3D volume of 64x64x64 voxels containing 12 "blobs"
blobs = Volume(dataurl + "blobs.vti")

isovalues = list(range(12))
iso_discrete = blobs.isosurface_discrete(isovalues, nsmooth=10)
iso_discrete.cmap(lut)

show(iso_discrete, __doc__, axes=1).close()
