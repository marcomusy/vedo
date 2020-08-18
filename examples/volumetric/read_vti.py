"""Load and render a 3D Volume

mode=0, composite rendering
mode=1, maximum-projection rendering"""
from vedo import datadir, load, show

vol1 = load(datadir+"vase.vti")

# can set colors and transparencies along the scalar range
# from minimum to maximum value. In this example voxels with
# the smallest value will be completely transparent (and white)
# while voxels with highest value of the scalar will get alpha=0.8
# and color will be=(0,0,1)
vol1.color(["white", "fuchsia", "dg", (0,0,1)])
#vol1.color('jet') # a matplotlib colormap name is also accepted
vol1.alpha([0.0, 0.2, 0.3, 0.8])

# a transparency for the GRADIENT of the scalar can also be set:
# in this case when the scalar is ~constant the gradient is ~zero
# and the voxel are made transparent:
vol1.alphaGradient([0.0, 0.5, 0.9])
vol1.addScalarBar3D(title='composite rendering', c='k').scale(0.8).x(20)

# mode = 1 is maximum-projection volume rendering
vol2 = load(datadir+"vase.vti").mode(1).addPos(60,0,0)
vol2.addScalarBar3D(title='maximum-projection', c='k').scale(0.8).x(160)

# show command creates and returns an instance of class Plotter
show(vol1, vol2, __doc__, axes=1)
