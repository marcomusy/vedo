"""Use sliders to slice a Volume
(click button to change colormap)"""
from vedo import dataurl, Volume, Text2D
from vedo.applications import Slicer3DPlotter

vol = Volume(dataurl + "embryo.slc")

plt = Slicer3DPlotter(
    vol,
    cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
    use_slider3d=False,
    bg="white",
    bg2="blue9",
)

# Can now add any other vedo object to the Plotter scene:
plt += Text2D(__doc__)

plt.show(viewup='z')
plt.close()
