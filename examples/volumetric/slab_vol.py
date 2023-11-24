"""Use slab() to extract a "thick" 2D slice from a 3D volume"""
from vedo import Axes, Volume, Box, dataurl, settings, show
from vedo.pyplot import histogram

settings.default_font = "Calco"

vol = Volume(dataurl + "embryo.tif")
vaxes = Axes(vol, xygrid=False)

slab = vol.slab([45,55], axis='z', operation='mean')
slab.cmap('Set1_r', vmin=10, vmax=80).add_scalarbar("intensity")
# histogram(slab).show().close()  # quickly inspect it

bbox = slab.metadata["slab_bounding_box"]
slab.z(-bbox[5] + vol.zbounds()[0])  # move slab to the bottom

# create a box around the slab for reference
slab_box = Box(bbox).wireframe().c("black")

show(__doc__, vol, slab, slab_box, vaxes, axes=14, viewup='z')
