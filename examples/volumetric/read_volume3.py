from vedo import Volume, dataurl
from vedo.applications import Slicer2DPlotter

vol = Volume(dataurl+"embryo.slc")

plt = Slicer2DPlotter(vol)

# use cmap("bw") to have black and white color scale
# no argument will grab the existing cmap in vol (or use build_lut())
# plt.cmap('bw').lighting(window=120, level=25)

plt.show().close()
