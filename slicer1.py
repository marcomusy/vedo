"""Use sliders to slice a Volume
(click button to change colormap)"""

from vedo import dataurl, Volume, Text2D, Sphere
from vedo.applications import Slicer3DPlotter

# Build the volumetric processing pipeline and render results.
vol = Volume(dataurl + "embryo.slc")

plt = Slicer3DPlotter(
    vol,
    cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
    use_slider3d=0,
    bg="white",
    bg2="blue9",
    N=2,
    sharecam=False,
    slider_positions=(
            ((0.35, 0.12), (0.475, 0.12)),
            ((0.35, 0.08), (0.475, 0.08)),
            ((0.35, 0.04), (0.475, 0.04)),
        ),
)

# Can now add any other vedo object to the Plotter scene:
plt += Text2D(__doc__)
plt.at(1).add(Text2D("Second renderer", c="k", font="Calco"))
plt.add(Sphere())

plt.show(viewup="z")
plt.interactive().close()
