# Perform other simple mathematical operation between 3d images.
# Possible operations are: +, -, /, 1/x, sin, cos, exp, log, abs, **2, sqrt, 
#   min, max, atan, atan2, median, mag, dot, gradient, divergence, laplacian.
# Alphas defines the opacity transfer function in the scalar range.
#
from vtkplotter import Plotter, vtkio,  utils
from vtkplotter.analysis import imageOperation

vp = Plotter(N=8, axes=4)

img0 = vtkio.loadImageData('data/embryo.slc')
v0 = utils.makeVolume(img0, c=0)
vp.show(v0, at=0)


img1 = imageOperation(img0, 'gradient')
img1 = imageOperation(img1, '+', 92.0)
v1 = utils.makeVolume(img1, c=1, alphas=[0,1,0,0,0])
vp.show(v1, at=1)

img2 = imageOperation(img0, 'divergence')
v2 = utils.makeVolume(img2, c=2)
vp.show(v2, at=2)

img3 = imageOperation(img0, 'laplacian')
v3 = utils.makeVolume(img3, c=3, alphas=[0,1,0,0,1])
vp.show(v3, at=3)

img4 = imageOperation(img0, 'median')
v4 = utils.makeVolume(img4, c=4)
vp.show(v4, at=4)

img5 = imageOperation(img0, 'sqrt')
v5 = utils.makeVolume(img5, c=5)
vp.show(v5, at=5)

img6 = imageOperation(img0, 'log')
v6 = utils.makeVolume(img6, c=6)
vp.show(v6, at=6)

img7 = imageOperation(img0, 'dot', img0)
v7 = utils.makeVolume(img7, c=7)
vp.show(v7, at=7, zoom=1.3, interactive=1)

