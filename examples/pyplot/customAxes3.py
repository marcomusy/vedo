"""Customizing Axes.
Cartesian planes can be displaced
from their lower-range default position"""
from vedo import Sphere, Axes, precision, show

sph = Sphere().scale([4,3,2]).shift(5,6,7).c('green2', 0.1).wireframe()

axs = Axes(sph,          # build axes for object sph
           xtitle="x axis",
           ytitle="y axis",
           ztitle="z axis",
           htitle='An ellipsoid at '+precision(sph.centerOfMass(),2),
           hTitleFont=1,
           hTitleColor='red3',
           zxGrid=True,
           xyFrameLine=2, yzFrameLine=2, zxFrameLine=2,
           xyFrameColor='red3',
           yzFrameColor='green3',
           zxFrameColor='blue3',
           xyShift=0.2,  # move xy plane 20% along z
           yzShift=0.2,  # move yz plane 20% along x
           zxShift=0.2,  # move zx plane 20% along y
)

show(sph, axs, __doc__).close()
