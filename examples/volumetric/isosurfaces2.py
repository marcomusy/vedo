"""Isosurface extraction from a volume dataset
with discrete values (labels)."""
from vedo import *

settings.default_font = "Antares"

lut = build_lut(
    [
        [0, "lightyellow"],
        [1, "red8"],
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
        [12, "red4"],
    ],
    interpolate=False,
)

# This dataset is a 3D volume of 64x64x64 voxels containing 12 "blobs"
blobs = Volume(dataurl + "blobs.vti")
box = blobs.box()

isovalues = list(range(1, 13))
iso_discrete = blobs.isosurface_discrete(
    isovalues,
    background_label=0,
    internal_boundaries=True,
    nsmooth=10,
)
iso_discrete.cmap(lut)

separate_blobs = []
txt_vols = []
for i in isovalues:
    b = iso_discrete.clone().threshold(0, i - 0.5, i + 0.5, on="cells")
    b.cap().clean().compute_normals()
    b.color(i).alpha(0.1).wireframe().lighting("off")
    v = b.volume() / 1e3
    cm = b.center_of_mass()
    txt = Text3D(f"blob {i}\nvol={v:.2f}", c=i, justify="center", pos=cm)
    txt_vols.append(txt)
    separate_blobs.append(b)

show([[iso_discrete, box, __doc__], [separate_blobs, txt_vols, box]], N=2, axes=1)

