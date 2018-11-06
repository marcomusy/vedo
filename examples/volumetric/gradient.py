# Calculate |gradient|, divergence, and laplacian of a voxel dataset
# alphas defines the opacity transfer function inthe scalar range.
#
from vtkplotter import Plotter, vtkio, analysis, utils

vp = Plotter(axes=4, shape=(1,4))

img0 = vtkio.loadImageData('data/embryo.slc')
v0 = utils.makeVolume(img0)

img1 = analysis.gradient(img0)
v1 = utils.makeVolume(img1, c='r', alphas=[0,1,0,0,0])

img2 = analysis.divergence(img0)
v2 = utils.makeVolume(img2, c='g')

img3 = analysis.laplacian(img0)
v3 = utils.makeVolume(img3, c='t', alphas=[0,1,0,0,1])

vp.show(v0, at=0)
vp.show(v1, at=1)
vp.show(v2, at=2)
vp.show(v3, at=3, zoom=1.3, interactive=1)

