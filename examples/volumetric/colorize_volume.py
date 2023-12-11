"""Custom color and transparency maps for Volumes"""
from vedo import Volume, dataurl, show
from vedo.pyplot import CornerHistogram

# Build a Volume object.
# A set of color/transparency values - of any length - can be passed
# to define the transfer function in the range of the scalar.
#  E.g.: setting alpha=[0, 0, 0, 1, 0, 0, 0] would make visible
#  only voxels with value close to center of the range (see histogram).
vol = Volume(dataurl + "embryo.slc")
vol.color(
    [
        (0, "green"),
        (49, "green"),
        (50, "blue"),
        (109, "blue"),
        (110, "red"),
        (180, "red"),
    ]
)
# vol.mode('max-projection')
vol.alpha([0.0, 1.0])
vol.alpha_unit(8)  # absorption unit, higher factors = higher transparency

vol.add_scalarbar3d(title="color:dot:alpha transfer function", c="k")
# substitute scalarbar3d to a 2d scalarbar
vol.scalarbar = vol.scalarbar.clone2d("center-right", 0.2)

ch = CornerHistogram(vol, logscale=True, pos="bottom-left")

# show both Volume and Mesh
show(vol, ch, __doc__, axes=1, zoom=1.2).close()
