from vedo import *

vol = Volume(dataurl+"embryo.slc").cmap('nipy_spectral')

vsl = VolumeSlice(vol) # reuse the same underlying data as in vol

# use colorize("bw") to have black and white color scale
# no argument will grab the existing cmap in vol (or use buildLUT())
vsl.colorize().lighting(window=100, level=25)

usage = Text2D(
    f"Image-style interactor:\n"
    f"SHIFT+Left click   \rightarrow rotate camera for oblique slicing\n"
    f"SHIFT+Middle click \rightarrow slice perpendicularly through image\n"
    f"Left click & drag  \rightarrow modify luminosity and contrast\n"
    f"X                  \rightarrow Reset to sagittal view\n"
    f"Y                  \rightarrow Reset to coronal view\n"
    f"Z                  \rightarrow Reset to axial view\n"
    f"R                  \rightarrow Reset the Window/Levels",
    font="Calco", pos="bottom-left", s=0.9, bg='yellow', alpha=0.25
)

custom_shape = [ # define here the 2 rendering rectangle spaces
    dict(bottomleft=(0.0,0.0), topright=(1,1), bg='k9'), # the full window
    dict(bottomleft=(0.7,0.7), topright=(1,1), bg='k8', bg2='lb'),
]

show([ (vsl,usage,"VolumeSlice example"), (vol,"Volume") ],
     shape=custom_shape,
     mode="image", bg='k9', zoom=1.2, axes=11, interactive=1).close()
